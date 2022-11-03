from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn.functional import gaussian_nll_loss

import pandas as pd

from ..utils.miscellaneous import assert_same_shapes


class Metrics:
    def __init__(
        self,
        df_preds: pd.DataFrame,
        metrics: List[str],
        methods: List[str],
        top: Optional[int] = None,
    ):
        self.df_preds = df_preds
        self.metrics = metrics
        self.methods = methods
        df = self._metrics_by(["method", "model"])
        if top is not None:
            self.df_preds = df_preds[
                df_preds.model.isin(
                    df.sort_values("nll").groupby("method").head(top).model
                )
            ]

    def _metrics_by(self, by: List[str]) -> pd.DataFrame:
        return (
            self.df_preds.groupby(by)
            .apply(compute_metrics)
            .apply(pd.Series)
            .reset_index()
        )[by + self.metrics]

    def _metrics_agg(
        self, df: pd.DataFrame, agg: Union[str, List[str]]
    ) -> pd.DataFrame:
        df = df.groupby(agg).agg(
            dict((metric, ["mean", "std"]) for metric in self.metrics)
        )
        df.columns = ["_".join(a) for a in df.columns.to_flat_index()]
        return df.sort_values("nll_mean")

    def by_method(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["method", "model"])
        if agg:
            return self._metrics_agg(df, "method").loc[self.methods]
        return df

    def by_dataset(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["dataset", "model"])
        if agg:
            return self._metrics_agg(df, "dataset")
        return df

    def by_unit(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["unit", "model"])
        if agg:
            return self._metrics_agg(df, "unit")
        return df

    def by_unit_method(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["method", "model", "unit"])
        if agg:
            return self._metrics_agg(df, ["method", "unit"])
        return df

    def by_best_model(self, metric: str = "nll") -> pd.DataFrame:
        df = self.by_method()
        return df.loc[df.reset_index().groupby("method")[metric].idxmin()]


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
