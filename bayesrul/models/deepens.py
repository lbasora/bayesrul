import random
from itertools import combinations
from typing import Iterator, List

import numpy as np
import pandas as pd


def deep_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    """
    df contains columns: preds, labels, stds for each base learner model,
    i.e. the test outputs of the base learner models
    """
    labels, preds, stds = None, [], []
    for _, model in df.groupby("model"):
        if labels is None:
            labels = model.labels.values
        preds.append(model.preds.values)
        stds.append(model.stds.values)

    mu_m = np.stack(preds)
    sigma_m = np.stack(stds)
    mu = mu_m.mean(axis=0)
    sigma = np.sqrt((mu_m**2 + sigma_m**2).mean(axis=0) - mu**2)
    return pd.DataFrame(
        dict(
            (k, v)
            for k, v in zip(["preds", "labels", "stds"], [mu, labels, sigma])
        )
    )


def deep_ensemble_gen(
    df: pd.DataFrame,
    base_learners: List[str],
    n_models_per_ens: int,
    max_deepens: int,
) -> Iterator[pd.DataFrame]:

    for method in base_learners:
        n = len(df.query(f"method=='{method}'").groupby("model"))
        comb = list(combinations(range(n), n_models_per_ens))
        for i, ens in enumerate(random.sample(comb, max_deepens)):
            models = []
            for model in ens:
                models.append(f"{method}_{model:03d}")
            yield deep_ensemble(df.query(f"model in {models}")).assign(
                method="DE", model=f"DE_{i:03d}"
            )
