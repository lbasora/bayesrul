import warnings
from typing import List, Optional, Dict, Tuple
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import seaborn as sns
import uncertainty_toolbox as uct

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
    aspect: Optional[int] = 1,
    label_rotation: Optional[int] = 0,
) -> sns.catplot:
    g = sns.catplot(
        x=x,
        y="value",
        hue="variable",
        data=df.reset_index()
        .assign(rmsce=lambda x: x.rmsce * 50)
        .melt(id_vars=x, value_vars=["rmse", "nll", "rmsce", "sharp"]),
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
        new_labels = ["RMSE", "NLL", "RMSCE*50", "Sharp"]
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
    xticklabel_size: int = 15,
    bbox: Tuple[float, float] = (0.45, 1.04),
):
    g = sns.catplot(
        x="fc",
        y="value",
        data=(
            df.melt(
                id_vars=["method", "fc"],
                value_vars=["nll", "rmse", "rmsce", "sharp"],
            ).assign(
                variable=lambda x: x.variable.map(
                    {
                        "nll": "NLL",
                        "rmse": "RMSE",
                        "rmsce": "RMSCE",
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
        aspect=1,
    ).set_titles(col_template="{col_name}", size=18)
    g.set(xlabel=None)
    g.set_xticklabels(size=xticklabel_size)
    g.set(xticklabels=[1, 2, 3], yticklabels=[])
    g.set(ylabel=None)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=bbox,
        ncol=6,
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize="18")
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_unit_method_metrics(
    df_unit_method: pd.DataFrame,
    save_as: Optional[str] = None,
) -> sns.catplot:
    g = (
        sns.catplot(
            x="unit",
            y="value",
            data=(
                df_unit_method.assign(rmsce=lambda x: x.rmsce * 50)
                .melt(
                    id_vars=["method", "unit"],
                    # value_vars=["rmse", "nll", "rmsce", "sharp"],
                    value_vars=["rmse", "rmsce", "sharp"],
                )
                .assign(
                    variable=lambda x: x.variable.map(
                        {"rmse": "RMSE", "rmsce": "RMSCE", "sharp": "SHARP"}
                    )
                )
            ),
            row="variable",
            hue="method",
            # margin_titles=True,
            kind="bar",
            sharey=False,
            aspect=5,
        )
        # .set_xticklabels(size=15)
        .set_titles(row_template="{row_name}", size=18).set(
            xlabel=None, ylabel="", yticks=[]
        )  # yscale="log"
    )
    leg = g._legend
    leg.set_bbox_to_anchor([0.95, 0.87])
    leg.set_title("")
    # new_labels = ["RMSE", "RMSCE*100", "Sharp"]
    # for t, l in zip(leg.texts, new_labels):
    #     t.set_text(l)
    plt.setp(leg.get_texts(), fontsize="18")
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def paired_column_facets(
    df: pd.DataFrame, y_vars: List, other_vars: Dict, **kwargs
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


def plot_unit_method_std_err(
    df: pd.DataFrame, methods: List[str], save_as: Optional[str] = None
) -> sns.FacetGrid:
    g = (
        paired_column_facets(
            df.query(f"method in {methods}"),
            y_vars=["stds_smooth", "errs_smooth"],
            other_vars={"x": "relative_time", "col": "method", "hue": "unit"},
            col_order=methods,
        )
        .set_titles(row_template="", col_template="{col_name}", size=18)
        .set(xlim=(0.0, 1.0))
    )
    for (row_name, _), ax in g.axes_dict.items():
        ax.set_ylabel(
            "Uncertainty RUL [cycles]"
            if row_name == "stds_smooth"
            else "Error RUL [cycles]",
        )
        ax.set_xlabel("Relative Lifetime")
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.08),
        ncol=11,
        # frameon=True,
        # markerscale=10,
    )
    leg = g._legend
    leg.set_title("")
    plt.setp(leg.get_texts(), fontsize="18")
    g.tight_layout()
    g.savefig(save_as)
    return g


def plot_rul(
    df: pd.DataFrame,
    methods: List[str],
    units: List[str],
    save_as: Optional[str] = None,
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
            "labels_smooth",
            color="black",
            label="True RUL",
            lw=2,
        )
        g.map(
            plt.plot,
            "relative_time",
            "preds_smooth",
            label="Predicted RUL",
            lw=2,
        )
        g = g.map(
            plt.fill_between,
            "relative_time",
            "preds_plus_smooth",
            "preds_minus_smooth",
            alpha=0.2,
            label="95\% CI",
        )
    g.set_titles(row_template="{row_name}", col_template="{col_name}", size=18)
    g.set_axis_labels("Relative Lifetime [-]", "RUL [cycles]")
    g.set(ylim=(0, 90))
    g.add_legend()
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.05),
        ncol=3,
    )
    plt.setp(g._legend.get_texts(), fontsize="15")
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_eps_al_uncertainty(
    df: pd.DataFrame,
    methods: List[str],
    units: List[str],
    save_as: Optional[str] = None,
) -> sns.relplot:
    g = (
        sns.relplot(
            data=df.query(f"method in {methods} and unit in {units}").melt(
                id_vars=["method", "unit", "relative_time"],
                value_vars=[
                    "stds_smooth",
                    "ep_stds_smooth",
                    "al_stds_smooth",
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
        .set_axis_labels("Relative Lifetime [-]", "RUL [cycles]")
        .set_titles(
            row_template="{row_name}", col_template="{col_name}", size=18
        )
    )
    for (row_name, _), ax in g.axes_dict.items():
        ax.set_ylabel("Uncertainty RUL (Std) [cycles]")
        ax.set_xlabel("Relative Lifetime")
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.45, 1.05),
        ncol=3,
    )
    leg = g._legend
    leg.set_title("")
    new_labels = [
        "Total uncertainty (std)",
        "Epistemic uncertainty (std)",
        "Aleatoric uncertainty (std)",
    ]
    for t, l in zip(leg.texts, new_labels):
        t.set_text(l)
    plt.setp(leg.get_texts(), fontsize="18")
    g.tight_layout()
    g.savefig(save_as)
    return g


def plot_calibration(
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


def plot_loss(
    df: pd.DataFrame,
    tags: List[str],
    xticks=None,
    log=True,
):
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
