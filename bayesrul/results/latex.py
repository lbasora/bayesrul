import pandas as pd


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
