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
