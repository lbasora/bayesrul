import subprocess
from typing import List

import pandas as pd


def df_to_latex(
    df: pd.DataFrame,
    dir: str,
    save_as: str,
    highlight_min: bool = True,
    pdf: bool = True,
) -> None:
    filename = f"{dir}/{save_as}.tex"
    pdffile = f"{save_as}.pdf"
    template = r"""\documentclass[preview]{{standalone}}
    \usepackage{{booktabs}}
    \begin{{document}}
    {}
    \end{{document}}
    """
    with open(filename, "wb") as f:
        table = latex_formatted(df, highlight_min)
        f.write(bytes(template.format(table), "UTF-8"))
    subprocess.call(
        [
            "pdflatex",
            "-output-directory",
            dir,
            "--interaction=batchmode",
            filename,
            "2>&1",
            " > ",
            "/dev/null",
        ]
    )


def latex_formatted(df: pd.DataFrame, highlight_min: bool = True) -> str:
    s = df.style
    if highlight_min:
        s = s.highlight_min(
            subset=df.columns[1:], props="textbf:--rwrap;", axis=0
        )
    s = s.format(precision=3)
    return (  # type: ignore
        s.hide(axis="index").to_latex(hrules=True).replace("_", " ")
    )


def to_mean_std(
    df: pd.DataFrame, metrics: List[str], precision: int = 3
) -> pd.DataFrame:
    df = df.assign(
        mae=lambda x: x.mae_mean.round(precision).astype(str)
        + " $\pm$ "
        + x.mae_std.round(precision).astype(str),
        rmse=lambda x: x.rmse_mean.round(precision).astype(str)
        + " $\pm$ "
        + x.rmse_std.round(precision).astype(str),
        nll=lambda x: x.nll_mean.round(precision).astype(str)
        + " $\pm$ "
        + x.nll_std.round(precision).astype(str),
        rmsce=lambda x: x.rmsce_mean.round(precision).astype(str)
        + " $\pm$ "
        + x.rmsce_std.round(precision).astype(str),
        sharp=lambda x: x.sharp_mean.round(precision).astype(str)
        + " $\pm$ "
        + x.sharp_std.round(precision).astype(str),
    )[["mae", "rmse", "nll", "rmsce", "sharp"]]
    return df
