import subprocess
from typing import List, Optional

import pandas as pd


def df_to_latex(
    df: pd.DataFrame,
    dir: str,
    save_as: str,
    highlight_min: bool = True,
) -> None:
    filename = f"{dir}/{save_as}.tex"
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


def latex_formatted(
    df: pd.DataFrame,
    highlight_min: bool = True,
) -> str:
    s = df.style
    if highlight_min:
        s = s.highlight_min(subset=df.columns[1:], props="textbf:--rwrap;", axis=0)
    s = s.format(precision=3)
    return (  # type: ignore
        s.hide(axis="index").to_latex(hrules=True).replace("_", " ")
    )


def to_mean_std(
    df: pd.DataFrame, precision: int = 3, metrics: List[str] = None
) -> pd.DataFrame:
    f = (
        lambda metric, x: x[f"{metric}_mean"].round(precision).astype(str)
        + " $\pm$ "
        + x[f"{metric}_std"].round(precision).astype(str)
    )
    d = {metric: f(metric, df) for metric in metrics}    
    return pd.concat([df[df.columns[0]], pd.DataFrame.from_dict(d)], axis=1)

