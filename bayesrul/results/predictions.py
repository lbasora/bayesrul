import logging
from pathlib import Path
from typing import List

from omegaconf import DictConfig

import numpy as np
import pandas as pd

from ..data.ncmapss.post_process import post_process, smooth_some_columns
from ..models.deepens import deep_ensemble_gen
from ..utils.miscellaneous import ResultSaver

log = logging.getLogger(__name__)


def load_predictions(
    methods: List[str],
    deepens: DictConfig,
    data_dir: str,
    runs_dir: str,
    cache_dir: str,
) -> pd.DataFrame:

    path = f"{cache_dir}/predictions.parquet"
    if Path(path).exists():
        return pd.read_parquet(path)

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
        f"Aggregating deep ensemble predictions for base learners {deepens.base_learners} ..."
    )
    df = pd.concat(cumul).reset_index(drop=True)
    # cumul = []
    # for base_learner in de_base_learners:
    #     de_df = deep_ensemble(df.query(f"method=='{base_learner}'"))
    #     de_df = post_process(de_df, data_path=data_dir, sigma=1.96).assign(
    #         method="DE", model=f"DE_{base_learner}"
    #     )
    #     cumul.append(de_df)
    cumul = [
        post_process(de_df, data_path=data_dir, sigma=1.96)
        for de_df in deep_ensemble_gen(
            df,
            deepens.base_learners,
            deepens.n_models_per_ens,
            deepens.max_deepens,
        )
    ]

    df = (
        pd.concat([df] + cumul)
        .reset_index(drop=True)
        .assign(
            errs=lambda x: x.preds - x.labels,
            ep_stds=np.sqrt(df.ep_vars),
            al_stds=np.sqrt(df.al_vars),
            dataset=lambda x: "D" + x.ds_id.astype(str),
            unit=lambda x: x.dataset + "U" + x.unit_id.map("{:02d}".format),
        )
    )
    Path(cache_dir).mkdir(exist_ok=True)
    df.to_parquet(f"{cache_dir}/predictions.parquet")
    return df


def smooth_cols(
    df_preds: pd.DataFrame, df_best_models: pd.DataFrame, cache_dir: str
) -> pd.DataFrame:

    path = f"{cache_dir}/predictions_best.parquet"
    if Path(path).exists():
        return pd.read_parquet(path)

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
        log.info(f"Smoothing columns for model {m} ...")
        if not Path(f"{cache_dir}/{m}.parquet").exists():
            smooth = (
                smooth_some_columns(
                    model,
                    smooth_cols,
                    bandwidth=bandwidths,
                ).assign(
                    ep_stds_smooth=model.ep_stds,
                    al_stds_smooth=model.al_stds,
                )
                if m.startswith("DE") or m.startswith("HNN")
                else smooth_some_columns(
                    model,
                    smooth_cols + ["ep_stds", "al_stds"],
                    bandwidth=bandwidths + [0.03, 0.03],
                )
            )
            smooth.to_parquet(f"{cache_dir}/{m}.parquet")

    df = pd.concat(
        [
            pd.read_parquet(f"{cache_dir}/{model}.parquet")
            for model in df_best_models.model.tolist()
            if Path(f"{cache_dir}/{model}.parquet").exists()
        ]
    )
    df.to_parquet(path)
    return df
