from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn.functional import gaussian_nll_loss

import pandas as pd

from ..utils.miscellaneous import assert_same_shapes


def model_metrics(
    df_preds: pd.DataFrame, metrics: List[str], top: Optional[int] = None
) -> pd.DataFrame:
    df_model = (
        df_preds.groupby(["method", "model"])
        .apply(compute_metrics)
        .apply(pd.Series)
        .reset_index()
    )
    if top is not None:
        df_preds = df_preds[
            df_preds.model.isin(
                df_model.sort_values("nll").groupby("method").head(top).model
            )
        ]
        df_model = (
            df_preds.groupby(["method", "model"])
            .apply(compute_metrics)
            .apply(pd.Series)
            .reset_index()
        )
    return df_model[["method", "model"] + metrics]


def method_metrics(df_model: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    df_method = df_model.groupby("method").agg(
        dict((metric, ["mean", "std"]) for metric in metrics)
    )
    df_method.columns = ["_".join(a) for a in df_method.columns.to_flat_index()]
    return df_method.sort_values("nll_mean")


def ds_metrics(
    df_preds: pd.DataFrame, metrics: List[str], agg: Optional[bool] = False
) -> pd.DataFrame:
    df_ds = (
        df_preds.groupby(["dataset", "model"])
        .apply(compute_metrics)
        .apply(pd.Series)
        .reset_index()[["dataset", "model"] + metrics]
    )
    if agg:
        df_ds = df_ds.groupby("dataset").agg(
            dict((metric, ["mean", "std"]) for metric in metrics)
        )
        df_ds.columns = ["_".join(a) for a in df_ds.columns.to_flat_index()]
        df_ds = df_ds.sort_values("nll_mean")
    return df_ds


def ds_unit(
    df_preds: pd.DataFrame, metrics: List[str], agg: Optional[bool] = False
) -> pd.DataFrame:
    pass


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    metrics = dict()
    y_true = torch.tensor(df["labels"].values, device=torch.device("cuda:0"))
    y_pred = torch.tensor(df["preds"].values, device=torch.device("cuda:0"))
    std = torch.tensor(df["stds"].values, device=torch.device("cuda:0"))
    metrics["mae"] = torch.abs(y_true - y_pred).mean().cpu().item()
    metrics["mse"] = ((y_true - y_pred) ** 2).mean().cpu().item()
    metrics["rmse"] = torch.sqrt(((y_true - y_pred) ** 2).mean()).cpu().item()
    metrics["sharp"] = sharpness(std).cpu().item()
    metrics["rmsce"] = rms_calibration_error(y_pred, std, y_true).cpu().item()
    metrics["nll"] = (
        gaussian_nll_loss(y_pred, y_true, torch.square(std)).cpu().item()
    )
    return metrics


def sharpness(sigma_hat: torch.Tensor) -> Tensor:
    sharp = torch.sqrt(torch.square(sigma_hat).mean())
    assert sharp.numel() == 1, f"Sharpness calculated is of shape {sharp.shape}"
    return sharp


def get_proportion_lists(
    y_pred: Tensor,
    y_std: Tensor,
    y_true: Tensor,
    num_bins: int,
    prop_type: Optional[str] = "interval",
) -> Tuple[Tensor, Tensor]:
    """To compute RMSCE"""
    exp_proportions = torch.linspace(0, 1, num_bins, device=y_true.get_device())

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(
        -1, 1
    )
    dist = Normal(  # type: ignore
        torch.tensor([0.0], device=y_true.get_device()),
        torch.tensor([1.0], device=y_true.get_device()),
    )
    if prop_type == "interval":
        gaussian_lower_bound = dist.icdf(0.5 - exp_proportions / 2.0)
        gaussian_upper_bound = dist.icdf(0.5 + exp_proportions / 2.0)

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound

        within_quantile = above_lower * below_upper
        obs_proportions = torch.sum(within_quantile, axis=0).flatten() / len(
            residuals
        )
    elif prop_type == "quantile":
        gaussian_quantile_bound = dist.icdf(exp_proportions)
        below_quantile = normalized_residuals <= gaussian_quantile_bound
        obs_proportions = torch.sum(below_quantile, axis=0).flatten() / len(
            residuals
        )

    return exp_proportions, obs_proportions


def rms_calibration_error(
    y_pred: Tensor,
    y_std: Tensor,
    y_true: Tensor,
    num_bins: Optional[int] = 100,
    prop_type: Optional[str] = "interval",
) -> Tensor:
    """RMSCE computation"""
    assert_same_shapes(y_pred, y_std, y_true)
    assert y_std.min() >= 0, "Not all values are positive"
    assert prop_type in ["interval", "quantile"]

    exp_props, obs_props = get_proportion_lists(
        y_pred, y_std, y_true, num_bins, prop_type
    )

    squared_diff_props = torch.square(exp_props - obs_props)
    rmsce = torch.sqrt(torch.mean(squared_diff_props))

    return rmsce
