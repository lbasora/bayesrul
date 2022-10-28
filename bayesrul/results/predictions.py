import logging
from pathlib import Path
from typing import List

import pandas as pd

from ..data.ncmapss.post_process import post_process, smooth_some_columns
from ..models.deepens import deep_ensemble
from ..utils.miscellaneous import ResultSaver

log = logging.getLogger(__name__)


def load_predictions(
    methods: List[str],
    de_base_learners: List[str],
    data_dir: str,
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
                df = post_process(
                    sav.load(), data_path=data_dir, sigma=1.96
                ).assign(model=model, method=method)
                cumul.append(df)

    log.info(
        f"Aggregating deep ensemble predictions for base learners {de_base_learners} ..."
    )
    df = pd.concat(cumul).reset_index(drop=True)
    cumul = []
    for base_learner in ["HNN"]:
        de_df = deep_ensemble(df.query(f"method=='{base_learner}'"))
        de_df = post_process(de_df, data_path=data_dir, sigma=1.96).assign(
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
    Path(cache_dir).mkdir()
    df.to_parquet(f"{cache_dir}/predictions.parquet")
    return df


def smooth_cols(
    df_preds: pd.DataFrame, df_best_models: pd.DataFrame, cache_dir: str
) -> None:
    smooth_cols = [
        "labels",
        "preds",
        "preds_plus",
        "preds_minus",
        "stds",
        "errs",
    ]
    bandwidths = [0.05, 0.01, 0.01, 0.01, 0.03, 0.03]
    for m, model in df_preds.groupby("model"):
        if m not in df_best_models.model.tolist():
            continue
        log.info(f"Soothing colums for model {m} ...")
        if not Path(f"{cache_dir}/{m}.parquet").exists():
            smooth = (
                smooth_some_columns(
                    model,
                    smooth_cols,
                    bandwidth=bandwidths,
                )
                # .assign(
                #     pred_std_smooth=model.pred_std,
                #      ep_std_smooth=model.ep_std,
                #      al_std_smooth=model.al_std,
                # )
                if m.startswith("DEEP_ENSEMBLE") or m.startswith("HETERO_NN")
                else smooth_some_columns(
                    model,
                    smooth_cols,  # + ["pred_std", "ep_std", "al_std"],
                    bandwidth=bandwidths,  # + [0.03, 0.03, 0.03],
                )
            )
            smooth.to_parquet(f"{cache_dir}/{m}.parquet")

    cumul = [
        pd.read_parquet(f"{cache_dir}/{model}.parquet")
        for model in df_best_models.model.tolist()
        if Path(f"{cache_dir}/{model}.parquet").exists()
    ]
    pd.concat(cumul).to_parquet(f"{cache_dir}/results_best.parquet")
