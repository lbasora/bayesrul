import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import uncertainty_toolbox as uct
from matplotlib.axes import Axes

import numpy as np
import pandas as pd

sns.set_style("white")
sns.set_context("paper")
sns.despine()
tex_fonts = {
    "text.usetex": True,
    "font.family": "Ubuntu",
    "axes.labelsize": 14,
    "font.size": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(tex_fonts)


def plot_metrics(
    df: pd.DataFrame,
    x: str,
    anchor: List[float],
    save_as: Optional[str] = None,
    legend: Optional[bool] = True,
    aspect: Optional[float] = 1.0,
    label_rotation: Optional[int] = 0,
) -> sns.catplot:
    g = sns.catplot(
        x=x,
        y="value",
        hue="variable",
        data=df.reset_index()
        .assign(ece=lambda x: x.ece * 50)
        .melt(id_vars=x, value_vars=["rmse", "nll", "ece", "sharp"]),
        kind="bar",
        aspect=aspect,
    )
    g.set(xlabel=None)
    g.set_xticklabels(rotation=label_rotation)
    g.set(yticklabels=[])
    g.set(ylabel=None)
    if legend:
        leg = g._legend
        leg.set_bbox_to_anchor(anchor)
        leg.set_title("")
        new_labels = ["RMSE", "NLL", "ECE*50", "Sharp"]
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)
    else:
        g._legend.remove()
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_fc_method_metrics(
    df: pd.DataFrame,
    methods: List[str],
    save_as: Optional[str] = None,
    xticklabel_size: int = 30,
    metric_label_size: int = 30,
    legend_label_size: int = 30,
    bbox: Tuple[float, float] = (0.45, 1.04),
):
    g = sns.catplot(
        x="fc",
        y="value",
        data=(
            df.melt(
                id_vars=["method", "fc"],
                value_vars=["nll", "rmse", "ece", "sharp"],
            ).assign(
                variable=lambda x: x.variable.map(
                    {
                        "nll": "NLL",
                        "rmse": "RMSE",
                        "ece": "ECE",
                        "sharp": "SHARP",
                    }
                )
            )
        ),
        col="variable",
        col_wrap=2,
        hue="method",
        hue_order=methods,
        kind="bar",
        sharey=False,
        aspect=1.5,
    ).set_titles(
        col_template="{col_name}",
        row_template="{row_name}",
        size=metric_label_size,
    )
    g.set(xlabel=None)
    g.set_xticklabels(size=xticklabel_size)
    g.set(xticklabels=["FC1", "FC2", "FC3"], yticklabels=[])
    g.set(ylabel=None)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=bbox,
        ncol=6,
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize=legend_label_size)
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_unit_method_metrics(
    df: pd.DataFrame,
    methods: List[str],
    xticklabel_size: int = 13,
    metric_label_size: int = 20,
    legend_label_size: int = 18,
    save_as=None,
) -> sns.FacetGrid:
    df = df.melt(
        id_vars=["method", "unit", "engine_id"],
        value_vars=["nll", "rmse", "ece", "sharp"],
    ).assign(
        theta=lambda x: x.engine_id * 2 * np.pi / len(df.unit.unique()),
        variable=lambda x: x.variable.apply(str.upper),
    )
    df = pd.concat(
        [
            df,
            df.query("engine_id==0").assign(
                theta=lambda x: x.theta + 2 * np.pi
            ),
        ]
    )
    df = df.sort_values(["variable", "method"]).reset_index(drop=True)
    metrics = ["RMSE", "NLL", "ECE", "SHARP"]
    g = sns.FacetGrid(
        data=df,
        col="variable",
        col_order=metrics,
        col_wrap=2,
        hue="method",
        hue_order=methods,
        subplot_kws=dict(projection="polar"),
        # aspect=2,
        height=6,
        sharex=False,
        sharey=False,
        despine=False,
    )
    g.map(sns.lineplot, "theta", "value", lw=3, errorbar=None)
    units = df.unit.unique()
    units = np.concatenate((units, [units[0]]))
    ticks = df.theta.unique()
    for ax, metric in zip(g.axes.flat, metrics):
        ax.xaxis.set_tick_params(pad=15)
        ax.set_xticks(ticks)
        ax.set_xticklabels(units, fontsize=xticklabel_size)
        angles = np.linspace(0, 2 * np.pi, len(ax.get_xticklabels()))
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles)
        for label, angle in zip(ax.get_xticklabels(), angles):
            x, y = label.get_position()
            lab = ax.text(
                x,
                y,
                label.get_text(),
                transform=label.get_transform(),
                ha=label.get_ha(),
                va=label.get_va(),
            )
            lab.set_rotation(angle)
        ax.set_xticklabels([])
        ax.set_title(metric, pad=60, fontdict={"fontsize": metric_label_size})

    plt.subplots_adjust(top=0.9)
    g.set(xlabel=None)
    g.set(yticklabels=[])
    g.set(ylabel=None)
    g.add_legend()
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.02),
        ncol=6,
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize=legend_label_size)
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_calibration_uct(
    df: pd.DataFrame,
    methods: List[str],
    save_as: Optional[str] = None,
) -> List[Axes]:
    _, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for i, method in enumerate(methods):
        pred_mean, pred_std, y = df.query(f"method=='{method}'")[
            ["preds", "stds", "labels"]
        ].T.to_numpy()
        ax = axs[i // 3, i % 3]
        uct.plot_calibration(pred_mean, pred_std, y, ax=ax)  # , n_subset=5000)
        ax.set_title(f"{method}", size=13)
    plt.savefig(save_as)
    return axs


def plot_calibration(
    df: pd.DataFrame, methods: List[str], save_as: Optional[str] = None
) -> sns.lineplot:
    g = sns.lineplot(
        x="exp_c",
        y="obs_c",
        hue="method",
        hue_order=["DE", "MCD", "HNN", "RAD", "LRT", "FO", "GT"],
        data=df,
        # ci=None,
        style="method",
        dashes={
            "LRT": (1, 0),
            "HNN": (1, 0),
            "DE": (1, 0),
            "FO": (1, 0),
            "RAD": (1, 0),
            "MCD": (1, 0),
            "GT": (10, 3),
        },
        palette={
            **{
                m: sns.color_palette().as_hex()[i]
                for i, m in enumerate(methods)
            },
            **{"GT": "0"},
        },
        lw=3,
        # ci=None,
    )
    g.set(
        xlabel="Expected Confidence Level", ylabel="Predicted Confidence Level"
    )
    leg = g.legend_
    leg.set_title("Method - Miscalibration Area")
    areas = df.groupby("method")["area"].mean()
    for t in leg.texts:
        if t.get_text() != "GT":
            t.set_text(f"{t.get_text()} - {areas[t.get_text()]:.2f}")
        else:
            t.set_text("Ideal calibration")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)


def plot_acc_vs_conf(
    df: pd.DataFrame, save_as: Optional[str] = None
) -> sns.lineplot:
    g = sns.lineplot(data=df, x="p", y="rmse", hue="method")
    g.set(xlabel="Confidence Level", ylabel="RMSE")
    g.legend_.set_title("")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    return g


def ood_boxplot(
    df: pd.DataFrame,
    save_as: Optional[str] = None,
    ylim: Optional[List[float]] = None,
    leg_labels: Optional[List[str]] = None,
) -> sns.boxplot:
    g = sns.boxplot(
        data=df,
        x="method",
        y="entropy",
        hue="ood_id",
        fliersize=0,
    )
    g.set(xlabel="Method", ylabel="Entropy")
    if ylim is not None:
        g.set_ylim(ylim)
    plt.legend(frameon=False)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=2,
    )
    leg = g.legend_
    leg.set_title("")
    if leg_labels is not None:
        for t, l in zip(leg.texts, leg_labels):
            t.set_text(l)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    return g


def ood_cdfplot(
    df: pd.DataFrame,
    methods: List[str],
    save_as: Optional[str] = None,
    xlim: Optional[List[float]] = None,
    leg_labels: Optional[List[str]] = ["ID", "OOD"],
) -> sns.lineplot:
    g = sns.lineplot(
        x="entropy",
        y="cdf",
        hue="method",
        hue_order=methods,
        style="ood_id",
        data=df,
        linewidth=3,
    )
    g.set(xlabel="Entropy", ylabel="CDF")
    if xlim is not None:
        g.set_xlim(xlim)
    leg_labels = [""] + methods + [""] + leg_labels
    leg = g.get_legend()
    for t, l in zip(leg.texts, leg_labels):
        t.set_text(l)
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    return g


def plot_rlt_metrics(
    df: pd.DataFrame,
    methods: List[str],
    save_as: Optional[str] = None,
    xticklabel_size: int = 13,
    metric_label_size: int = 18,
    legend_label_size: int = 18,
) -> sns.catplot:
    g = sns.catplot(
        x="relative_time",
        y="value",
        data=(
            df.melt(
                id_vars=["method", "relative_time"],
                value_vars=["nll", "rmse", "ece", "sharp"],
            ).assign(
                variable=lambda x: x.variable.map(
                    {
                        "nll": "NLL",
                        "rmse": "RMSE",
                        "ece": "ECE",
                        "sharp": "SHARP",
                    }
                )
            )
        ),
        col="variable",
        col_wrap=2,
        hue="method",
        hue_order=methods,
        kind="point",
        sharey=False,
        aspect=2,
        height=3,
    )
    g.set_titles(col_template="{col_name}", size=metric_label_size)
    g.set(xlabel="Relative Lifetime")
    g.set(yticklabels=[])
    g.set(ylabel=None)
    g.set_xticklabels(size=xticklabel_size)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.02),
        ncol=6,
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize=legend_label_size)
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def paired_column_facets(
    df: pd.DataFrame, y_vars: List[str], other_vars: Dict[str, str], **kwargs
) -> sns.FacetGrid:
    g = df.melt(list(other_vars.values()), y_vars).pipe(
        (sns.relplot, "data"),
        **other_vars,
        y="value",
        size="unit",
        row="variable",
        facet_kws=dict(sharey="row", margin_titles=True),
        palette="Spectral",
        kind="line",
        **kwargs,
    )
    return g


def plot_rlt_unit_method_std_err(
    df: pd.DataFrame,
    methods: List[str],
    units: List[str],
    grey_units: List[str],
    save_as: Optional[str] = None,
    xticklabel_size: int = 18,
    method_label_size: int = 30,
    legend_label_size: int = 25,
) -> sns.FacetGrid:
    hue_order = (
        df.query("unit in @units")
        .groupby("unit")
        .mean(numeric_only=True)
        .sort_values("stds", ascending=False)
        .reset_index()
        .unit.unique()
        .tolist()
    )
    g = (
        paired_column_facets(
            df.query("unit in @units").query(f"method in {methods}"),
            y_vars=["stds", "errs"],
            other_vars={"x": "relative_time", "col": "method", "hue": "unit"},
            col_order=methods,
            hue_order=hue_order,
            sizes=(5, 10),
            markers=True,
        )
        .set_titles(
            row_template="",
            col_template="{col_name}",
            size=method_label_size,
        )
        .set(xlim=(0.0, 1.0))
    )
    for (row_name, col_name), ax in g.axes_dict.items():
        ax.set_ylabel(
            "Uncertainty RUL [cycles]"
            if row_name == "stds"
            else "Error RUL [cycles]",
            fontsize=25,
        )
        ax.set_xlabel("Relative Lifetime", fontsize=25)
        ax.xaxis.set_tick_params(labelsize=xticklabel_size)
        ax.yaxis.set_tick_params(labelsize=xticklabel_size)
        data = (
            df.query("unit in @grey_units and method==@col_name")
            .groupby(["method", "unit", "relative_time"])
            .mean(numeric_only=True)
            .reset_index()[["relative_time", "unit", "errs", "stds"]]
        )
        sns.lineplot(
            data=data,
            x="relative_time",
            y=row_name,
            units="unit",
            estimator=None,
            color=".7",
            linewidth=1,
            legend=None,
            errorbar=None,
            ax=ax,
        )
    sns.move_legend(
        g, "upper center", bbox_to_anchor=(0.45, 1.08), ncol=11, handlelength=6
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize=legend_label_size)
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_rlt_unit_method_rul(
    df: pd.DataFrame,
    methods: List[str],
    units: List[str],
    save_as: Optional[str] = None,
    xticklabel_size: int = 18,
    method_label_size: int = 20,
    legend_label_size: int = 18,
) -> sns.FacetGrid:
    g = sns.FacetGrid(
        df.query(f"method in {methods} and unit in {units}"),
        row="unit",
        col="method",
        col_order=methods,
        margin_titles=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        g.map(
            plt.plot,
            "relative_time",
            "labels",
            color="black",
            label="True RUL",
            lw=2,
        )
        g.map(
            plt.plot,
            "relative_time",
            "preds",
            label="Predicted RUL",
            lw=2,
        )
        g = g.map(
            plt.fill_between,
            "relative_time",
            "preds_plus",
            "preds_minus",
            alpha=0.2,
            label="95\% CI",
        )
    g.set_titles(
        row_template="{row_name}",
        col_template="{col_name}",
        size=method_label_size,
    )
    for ax in g.axes.flat:
        ax.xaxis.set_tick_params(labelsize=xticklabel_size)
        ax.yaxis.set_tick_params(labelsize=xticklabel_size)
    g.set_axis_labels(
        "Relative Lifetime [-]", "RUL [cycles]", size=xticklabel_size
    )
    g.set(ylim=(0, 90))
    g.add_legend()
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.05),
        ncol=3,
    )
    plt.setp(g._legend.get_texts(), fontsize=legend_label_size)
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_eps_al_uncertainty(
    df: pd.DataFrame,
    methods: List[str],
    units: List[str],
    save_as: Optional[str] = None,
    xticklabel_size: int = 18,
    method_label_size: int = 20,
    legend_label_size: int = 18,
) -> sns.relplot:
    g = (
        sns.relplot(
            data=df.query(f"method in {methods} and unit in {units}").melt(
                id_vars=["method", "unit", "relative_time"],
                value_vars=[
                    "stds",
                    "ep_stds",
                    "al_stds",
                ],
            ),
            kind="line",
            x="relative_time",
            y="value",
            hue="variable",
            row="unit",
            col="method",
            col_order=methods,
            facet_kws=dict(sharey="row", margin_titles=True),
            linewidth=3,
        )
        .set_axis_labels(
            "Relative Lifetime [-]", "RUL [cycles]", size=xticklabel_size
        )
        .set_titles(
            row_template="{row_name}",
            col_template="{col_name}",
            size=method_label_size,
        )
    )
    for (row_name, _), ax in g.axes_dict.items():
        ax.set_ylabel("Std [cycles]")
        ax.set_xlabel("Relative Lifetime [-]")
        ax.xaxis.set_tick_params(labelsize=xticklabel_size)
        ax.yaxis.set_tick_params(labelsize=xticklabel_size)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.05),
        ncol=3,
    )
    leg = g._legend
    leg.set_title("")
    new_labels = [
        "Total (std)",
        "Epistemic (std)",
        "Aleatoric (std)",
    ]
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.setp(leg.get_texts(), fontsize=legend_label_size)
    g.tight_layout()
    g.savefig(save_as)
    return g


def plot_loss(
    df: pd.DataFrame,
    tags: List[str],
    xticks=None,
    log=True,
) -> sns.relplot:
    g = (
        sns.relplot(
            data=df,
            x="epoch",
            y="value",
            hue="method",
            col="tag",
            # col_wrap=2,
            kind="line",
            col_order=tags,
            height=3,
            aspect=0.9,
            facet_kws=dict(sharey=False),
        )
        .set_titles(row_template="", col_template="{col_name}", size=12)
        .set(ylabel=None, xticks=xticks)
    )
    if log:
        g.set(yscale="log")
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.02),
        ncol=3,
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize="12")
    return g
