import logging
from pathlib import Path
from typing import List

import pandas as pd

from ..data.ncmapss.post_process import post_process
from ..models.deepens import deep_ensemble
from ..utils.miscellaneous import ResultSaver

log = logging.getLogger(__name__)


def load_predictions(
    methods: List[str],
    de_base_learners: List[str],
    runs_dir: str,
    cache_dir: str,
) -> pd.DataFrame:

    if Path(cache_dir).exists():
        return pd.read_parquet(f"{cache_dir}/predictions.parquet")

    log.info(f"Loading predictions from {runs_dir} ...")
    cumul = []
    for method in methods:
        for p in Path(runs_dir).glob(f"{method}/*"):
            if p.joinpath(f"{p}/predictions/results.parquet").exists():
                model = f"{method}_{int(p.stem):03d}"
                log.info(f"Loading predictions for model {model} ...")
                sav = ResultSaver(p)
                df = post_process(sav.load(), sigma=1.96).assign(
                    model=model, method=method
                )
                cumul.append(df)

    log.info(
        f"Aggregating deep ensemble predictions for base learners {de_base_learners} ..."
    )
    df = pd.concat(cumul).reset_index(drop=True)
    cumul = []
    for base_learner in ["HNN", "FO"]:
        de_df = deep_ensemble(df.query(f"method=='{base_learner}'"))
        de_df = post_process(de_df, sigma=1.96).assign(
            method="DE", model=f"DE_{base_learner}"
        )
        cumul.append(de_df)

    df = (
        pd.concat([df] + cumul)
        .reset_index(drop=True)
        .assign(
            errs=lambda x: x.preds - x.labels,
            # pred_std=np.sqrt(df.pred_var),
            # ep_std=np.sqrt(df.ep_var),
            # al_std=np.sqrt(df.al_var),
            dataset=lambda x: "D" + x.ds_id.astype(str),
            unit=lambda x: x.dataset + "U" + x.unit_id.map("{:02d}".format),
        )
    )
    df.to_parquet(f"{cache_dir}/predictions.parquet")
    return df
