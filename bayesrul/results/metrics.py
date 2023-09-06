from typing import Dict, List, Optional, Tuple, Union

import scipy
import torch
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from torch import Tensor
from torch.distributions import Normal
from torch.nn.functional import gaussian_nll_loss
from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists_vectorized,
)

import numpy as np
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
            return self._metrics_agg(df, "method")  # .loc[self.methods]
        return df

    def by_dataset(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["dataset", "model"])
        if agg:
            return self._metrics_agg(df, "dataset")
        return df

    def by_fc(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["fc", "model"])
        if agg:
            return self._metrics_agg(df, "fc")
        return df

    def by_fc_method(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["method", "model", "fc"])
        if agg:
            return self._metrics_agg(df, "fc")
        return df

    def by_unit(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["unit", "model"])
        if agg:
            return self._metrics_agg(df, "unit")
        return df

    def by_unit_method(self, agg: bool = False) -> pd.DataFrame:
        df = self._metrics_by(["method", "model", "unit", "engine_id"])
        if agg:
            return self._metrics_agg(df, ["method", "unit"])
        return df

    def by_best_model(self, metric: str = "nll") -> pd.DataFrame:
        df = self.by_method()
        return df.loc[df.reset_index().groupby("method")[metric].idxmin()]

    def calibration(self, fc: Optional[int] = None) -> pd.DataFrame:
        if fc is not None:
            return calibration(self.df_preds.query(f"fc=={fc}"))
        return calibration(self.df_preds)

    def acc_vs_conf(self) -> pd.DataFrame:
        return acc_vs_conf(self.df_preds)

    def by_rlt_method(self, rlt_round: Optional[int] = 1) -> pd.DataFrame:
        self.df_preds = self.df_preds.assign(
            relative_time=lambda x: x.relative_time.round(rlt_round)
        )
        return self._metrics_by(["relative_time", "method", "model"])

    def by_rlt_unit_method_std_err(
        self, rlt_round: Optional[int] = 1, ewm_span: Optional[int] = None
    ) -> pd.DataFrame:
        return (
            self.df_preds.assign(
                relative_time=lambda x: x.relative_time.round(rlt_round)
            )
            .groupby(["relative_time", "method", "model", "unit"])
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("relative_time")
        )[["relative_time", "method", "model", "unit", "errs", "stds"]]

    def by_rlt_unit_method_rul(
        self, rlt_round: Optional[int] = 2, ewm_span: Optional[int] = 3
    ) -> pd.DataFrame:
        return (
            self.df_preds.assign(
                relative_time=lambda x: x.relative_time.round(rlt_round)
            )
            .groupby(["relative_time", "method", "model", "unit"])
            .mean(numeric_only=True)
            .ewm(span=ewm_span)
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("relative_time")
        )[
            [
                "relative_time",
                "method",
                "model",
                "unit",
                "preds",
                "labels",
                "preds_plus",
                "preds_minus",
            ]
        ]

    def nasa_score(self, agg: bool = False) -> pd.DataFrame:
        return self.by_method(
            agg
        )  # pd.concat(list(self.by_method(agg)))  # , self.by_dataset(agg)

    def by_rlt_unit_method_eps_al(
        self, rlt_round: Optional[int] = 2, ewm_span: Optional[int] = 5
    ) -> pd.DataFrame:
        return (
            self.df_preds.assign(
                relative_time=lambda x: x.relative_time.round(rlt_round)
            )
            .groupby(["relative_time", "method", "model", "unit"])
            .mean(numeric_only=True)
            .ewm(span=ewm_span)
            .mean(numeric_only=True)
            .reset_index()
            .sort_values("relative_time")
        )[
            [
                "relative_time",
                "method",
                "model",
                "unit",
                "preds",
                "labels",
                "stds",
                "ep_stds",
                "al_stds",
            ]
        ]


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
    metrics["ece"] = (
        mean_absolute_calibration_error(y_pred, std, y_true).cpu().item()
    )
    metrics["nll"] = (
        gaussian_nll_loss(y_pred, y_true, torch.square(std)).cpu().item()
    )
    metrics["entropy"] = lambda x: torch.distributions.normal.Normal(
        torch.tensor(x.labels.values), torch.tensor(x.stds.values)
    ).entropy()
    metrics["s"] = nasa_score(y_true, y_pred).mean().cpu().item()
    return metrics


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return torch.where(d > 0, torch.exp(d / 10) - 1, torch.exp(-d / 13) - 1)


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


def mean_absolute_calibration_error(
    y_pred: Tensor,
    y_std: Tensor,
    y_true: Tensor,
    num_bins: int = 100,
    prop_type: str = "interval",
) -> float:
    """Mean absolute calibration error; identical to ECE."""
    assert_same_shapes(y_pred, y_std, y_true)
    assert y_std.min() >= 0, "Not all values are positive"
    assert prop_type in ["interval", "quantile"]

    # Get lists of expected and observed proportions for a range of quantiles
    exp_props, obs_props = get_proportion_lists(
        y_pred, y_std, y_true, num_bins, prop_type
    )

    abs_diff_proportions = torch.abs(exp_props - obs_props)
    mace = torch.mean(abs_diff_proportions)

    return mace


def miscalibration_area(
    exp_proportions: np.array, obs_proportions: np.array
) -> np.array:
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    return np.asarray(polygon_area_list).sum()


def calibration(df: pd.DataFrame) -> pd.DataFrame:
    cumul = []
    for (method, model), df_model in df.groupby(["method", "model"]):
        expected_p, observed_p = get_proportion_lists_vectorized(
            df_model["preds"].values,
            df_model["stds"].values,
            df_model["labels"].values,
        )
        area = miscalibration_area(expected_p, observed_p)
        cumul.append((method, model, expected_p, observed_p, area))
        # break
    cumul.append(("GT", "GT", expected_p, expected_p, 0))
    df_cal = (
        pd.DataFrame.from_records(
            cumul, columns=["method", "model", "exp_c", "obs_c", "area"]
        )
        .explode(["exp_c", "obs_c"])
        .reset_index(drop=True)
    )
    return df_cal


def acc_vs_conf(df: pd.DataFrame) -> pd.DataFrame:
    percentiles = np.arange(100) / 100.0
    cumul = []
    for _, df_model in df.groupby(["method", "model"]):
        df_model = df_model.sort_values("stds", ascending=False)
        cutoff_inds = (percentiles * df_model.shape[0]).astype(int)
        for cutoff, p in zip(cutoff_inds, percentiles):
            cumul.append(
                df_model[cutoff:]
                .assign(rmse=lambda x: (np.sqrt((x.labels - x.preds) ** 2)))[
                    ["method", "model", "rmse"]
                ]
                .groupby(["method", "model"])
                .mean()
                .assign(p=p)
            )
    return pd.concat(cumul).reset_index()


def compute_cdf(df: pd.DataFrame) -> pd.DataFrame:
    cumul = []
    unc_ = np.linspace(df["entropy"].min(), df["entropy"].max(), 100)
    for (method, model), df_model in df.groupby(["method", "model"]):
        for ood_id in ["OOD", "ID"]:
            df_ = df_model.query(f"ood_id=='{ood_id}'")
            unc = np.sort(df_["entropy"])
            prob = np.linspace(0, 1, unc.shape[0])
            f_cdf = scipy.interpolate.interp1d(
                unc, prob, fill_value=(0.0, 1.0), bounds_error=False
            )
            prob_ = f_cdf(unc_)
            cumul.append((method, model, unc_, prob_, ood_id))

    return (
        pd.DataFrame.from_records(
            cumul, columns=["method", "model", "entropy", "cdf", "ood_id"]
        )
        .explode(["entropy", "cdf"])
        .reset_index(drop=True)
    )
