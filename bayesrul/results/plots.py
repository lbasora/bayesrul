from typing import List, Optional, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import uncertainty_toolbox as uct

import pandas as pd

sns.set_style("white")
sns.set_context("paper")
sns.despine()
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
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
    g = (
        sns.catplot(
            x=x,
            y="value",
            hue="variable",
            data=df.reset_index()
            .assign(rmsce=lambda x: x.rmsce * 100)
            .melt(id_vars=x, value_vars=["rmse", "nll", "rmsce", "sharp"]),
            kind="bar",
            aspect=aspect,
        )
        .set_xticklabels(rotation=label_rotation)
        .set(xlabel=None, ylabel="", yscale="log", yticklabels=[])
    )
    if legend:
        leg = g._legend
        leg.set_bbox_to_anchor(anchor)
        leg.set_title("")
        new_labels = ["RMSE", "NLL", "RMSCE*100", "Sharp"]
        for t, l in zip(leg.texts, new_labels):
            t.set_text(l)
    else:
        g._legend.remove()
    g.tight_layout()
    if save_as is not None:
        g.savefig(save_as)
    return g


def plot_unit_method_metrics(
    df_unit_method: pd.DataFrame,
    save_as: Optional[str] = None,
) -> sns.catplot:
    sns.set_style("white")
    sns.set_context("paper")
    sns.despine()
    g = (
        sns.catplot(
            x="unit",
            y="value",
            data=(
                df_unit_method.assign(rmsce=lambda x: x.rmsce * 100)
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


def plot_unit_method_std_err(df, methods, save_as):
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
    g.savefig(f"figs/{save_as}.pdf")
    return g
