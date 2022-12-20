from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from typing import Union, List, Dict


class ResultSaver:
    def __init__(self, path: Union[Path, str], filename: str) -> None:
        self.path = Path(path)
        self.path.mkdir(exist_ok=True)
        self.file_path = Path(self.path, filename)

    def save(self, df: pd.DataFrame) -> None:
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        assert isinstance(df, pd.DataFrame), f"{type(df)} is not a dataframe"
        df.to_parquet(self.file_path)

    def load(self) -> pd.DataFrame:
        return pd.read_parquet(self.file_path)

    def append(
        self, series: Union[List[pd.Series], Dict[str, np.array]]
    ) -> None:
        if isinstance(series, list):
            series = pd.concat(series, axis=1)
        if isinstance(series, dict):
            series = pd.DataFrame(series)
        df = self.load()
        df = pd.concat([df, series], axis=1)
        assert isinstance(df, pd.DataFrame), f"{type(df)} is not a dataframe"
        s = df.isna().sum()
        if isinstance(s, pd.Series):
            s = s.sum()
        assert s == 0, "NaNs introduced in results dataframe"
        self.save(df)


def assert_same_shapes(*args):
    assert len(args) > 1, "Needs to be provided more than one argument"
    shape = args[0].shape
    for arr in args[1:]:
        assert arr.shape == shape

    return True


def weights_init(m):
    """Initializes weights of a nn.Module : xavier for conv
    and kaiming for linear

    """
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    assert isinstance(m, torch.nn.Module), f"{type(m)} is not a nn.Module"
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
