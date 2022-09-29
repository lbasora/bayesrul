import glob
import os
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import rank_zero_only

import numpy as np


class Dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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


def get_checkpoint(path: Union[Path, str], version=None) -> Union[str, None]:
    """Gets the checkpoint filename and path of a log directory"""
    try:
        path = os.path.join(os.getcwd(), path, "lightning_logs")
        ls = sorted(os.listdir(path), reverse=True)
        d = os.path.join(path, ls[-1], "checkpoints")
        if os.path.isdir(d):
            checkpoint_file = sorted(
                glob.glob(os.path.join(d, "*.ckpt")),
                key=os.path.getmtime,
                reverse=True,
            )
            return str(checkpoint_file[0]) if checkpoint_file else None
        return None
    except Exception as e:
        if e == FileNotFoundError:
            print("Could not find any checkpoint in {}".format(d))
        return None


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


def simple_cull(inputPoints):
    def dominates(row, candidateRow):
        return sum([row[x] <= candidateRow[x] for x in range(len(row))]) == len(
            row
        )

    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break

    return paretoPoints, dominatedPoints


def select_pareto(df, paretoSet):
    """Selection of pareto optimal set"""
    arr = np.array([[x1, x2] for x1, x2 in list(paretoSet)])
    return df[(df.values_0.isin(arr[:, 0])) & (df.values_1.isin(arr[:, 1]))]
