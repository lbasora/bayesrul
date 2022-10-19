"""
Metrics for evaluation on the test set
"""

import torch
from torch.distributions import Normal
from bayesrul.utils.miscellaneous import assert_same_shapes


def sharpness(sigma_hat):
    sharp = torch.sqrt(torch.square(sigma_hat).mean())
    assert sharp.numel() == 1, f"Sharpness calculated is of shape {sharp.shape}"
    return sharp


def get_proportion_lists(y_pred, y_std, y_true, num_bins, prop_type="interval"):
    """To compute RMSCE"""
    exp_proportions = torch.linspace(0, 1, num_bins, device=y_true.get_device())

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(
        -1, 1
    )
    dist = Normal(
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
    y_pred, y_std, y_true, num_bins=100, prop_type="interval"
):
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
