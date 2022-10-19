import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Union

import h5py
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import pandas as pd

from ..lmdb_utils import StandardScalerTransform, create_lmdb, make_slice
from .dataset import NCMAPSSDataModule

log = logging.getLogger(__name__)


def build_ds(cfg: DictConfig):
    generate_lmdb(cfg)
    preprocess_lmdb(cfg)


def get_features(cfg: DictConfig):
    features = [
        cfg.dataset.features[key]
        for key in cfg.dataset.features.keys()
        if (key != "Y") and (key in cfg.dataset.subdata)
    ]
    return [f for fl in features for f in fl if f]


def generate_lmdb(cfg: DictConfig) -> None:
    """Parquet files to lmdb files"""
    lmdb_dir = Path(f"{cfg.paths.output_dir}/lmdb")
    lmdb_dir.mkdir(exist_ok=True)
    lmdb_dir_files = [x for x in lmdb_dir.iterdir()]
    if len(lmdb_dir_files) > 0:
        log.warning(
            f"{lmdb_dir} is not empty. Generation will not overwrite"
            " the previously generated .lmdb files. It will append data."
        )
    features = get_features(cfg)
    for ds in ["train", "test"]:
        filelist = list(
            Path(cfg.dataset.hdf5_path, filename)
            for filename in cfg.dataset.files
        )
        if filelist is not None:
            log.info(
                f"Generating {ds} lmdb with {[x.stem for x in filelist]} files..."
            )
            iterator = process_files(
                filelist,
                "dev" if ds == "train" else "test",
                cfg.dataset.bits,
                cfg.dataset.win_length,
                cfg.dataset.win_step,
                cfg.dataset.files,
                features,
                cfg.dataset.subdata,
                cfg.dataset.skip_obs,
            )
            feed_lmdb(
                Path(f"{lmdb_dir}/{ds}.lmdb"),
                iterator,
                cfg.dataset.bits,
                cfg.dataset.win_length,
                len(features),
            )


class Line(NamedTuple):  # An N-CMAPSS Line
    ds_id: int  # Which ds
    unit_id: int  # Which unit
    win_id: int  # Window id
    data: np.ndarray  # X_s, X_v, T, A (not necessarily all of them)
    rul: int  # Y


def process_files(
    filelist: List[Path],
    dev_test: str,
    bits: int,
    win_length: int,
    win_step: int,
    files: List[str],
    features: List[str],
    subdata: List[str],
    skip_obs: Optional[int] = None,
) -> Iterator[Line]:

    for filename in tqdm(filelist):
        df = read_hdf5(
            filename,
            dev_test,
            np.float32 if bits == 32 else np.float64,
            files,
            vars=subdata,
        )
        if "A" in subdata:
            df = linear_piece_wise_RUL(df.copy())
        if skip_obs is not None:
            df = df[::skip_obs]
        ds_id = int(str(filename.stem).split("_")[-1].split("-")[0][3])
        yield from process_dataframe(df, ds_id, win_length, win_step, features)
        del df


def read_hdf5(
    filepath: Path,
    dev_test_suffix: str,
    dtype: Union[np.float32, np.float64],
    files: List[str],
    vars=["W", "X_s", "X_v", "T", "A"],
):
    """Load data from source file into a dataframe.

    Parameters
    ----------
    file : str
        Source file.
    vars : list
        May contain 'X_s', 'X_v', 'T', 'A'
        W: Scenario Descriptors (always included)
        X_s: Measurements
        X_v: Virtual sensors
        T: Health Parameters
        A: Auxiliary Data
        Y: rul (always included)

    Returns
    -------
    DataFrame
        Data organized into a dataframe.
    """

    assert all(
        [x in ["W", "X_s", "X_v", "T", "A"] for x in vars]
    ), "Wrong vars provided, choose a subset of ['W', 'X_s', 'X_v', 'T', 'A']"
    assert filepath.name in files, "Incorrect file name {}".format(filepath)

    if ".h5" not in filepath.name:
        filepath = filepath.with_suffix(".h5")

    with h5py.File(filepath, "r") as hdf:
        data = []
        varnames = []

        data.append(np.array(hdf.get(f"W_{dev_test_suffix}")))
        varnames.extend(hdf.get("W_var"))

        if "X_s" in vars:
            data.append(np.array(hdf.get(f"X_s_{dev_test_suffix}")))
            varnames.extend(hdf.get("X_s_var"))
        if "X_v" in vars:
            data.append(np.array(hdf.get(f"X_v_{dev_test_suffix}")))
            varnames.extend(hdf.get("X_v_var"))
        if "T" in vars:
            data.append(np.array(hdf.get(f"T_{dev_test_suffix}")))
            varnames.extend(hdf.get("T_var"))
        if "A" in vars:
            data.append(np.array(hdf.get(f"A_{dev_test_suffix}")))
            varnames.extend(hdf.get("A_var"))

        # Add RUL
        data.append(np.array(hdf.get(f"Y_{dev_test_suffix}")))

    varnames = list(np.array(varnames, dtype="U20"))  # Strange string types
    varnames = [str(x) for x in varnames]
    varnames.append("rul")

    return pd.DataFrame(
        data=np.concatenate(data, axis=1), columns=varnames, dtype=dtype
    )


def linear_piece_wise_RUL(df: pd.DataFrame, drop_hs=True) -> pd.DataFrame:
    """Corrects the RUL label. Uses the Health State to change RUL
        into piece-wise linear RUL (reduces overfitting on the healthy part)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to correct.

    Returns
    -------
    df : pd.DataFrame
        Corrected dataframe

    Raises
    ------
    KeyError:
        When 'A' subset is not selected and extract_validation bypassed
    """
    healthy = df[["unit", "hs", "rul"]]  # to reduce memory footprint
    healthy = df[df.hs == 1]  # Filter on healthy

    mergedhealthy = healthy.merge(
        healthy.groupby(["unit", "hs"])["rul"].min(),
        how="inner",
        on=["unit", "hs"],
    )  # Compute the max linear rul
    df = df.merge(mergedhealthy, how="left", on=list(df.columns[:-1])).drop(
        columns=["rul_x"]
    )  # Put it back on original dataframe
    df.rul_y.fillna(df["rul"], inplace=True)  # True RUL values on NaNs
    df.drop(columns=["rul"], inplace=True)
    df.rename({"rul_y": "rul"}, axis=1, inplace=True)

    assert df.isna().sum().sum() == 0, "NaNs in df, on columns {}".format(
        df.isna().sum()[df.isna().sum() >= 1].index.values.tolist()
    )

    if drop_hs:
        df.drop(columns=["hs"], inplace=True)

    return df


def process_dataframe(
    df: pd.DataFrame,
    ds_id: int,
    win_length: int,
    win_step: int,
    features: List[str],
) -> Iterator[Line]:

    pbar_unit = tqdm(df.groupby("unit"), leave=False, position=1)
    for unit_id, traj in pbar_unit:
        pbar_unit.set_description(f"Unit {int(unit_id)}")
        pbar_win = tqdm(
            make_slice(traj.shape[0], win_length, win_step),
            leave=False,
            total=traj.shape[0] / win_step,
            position=2,
        )
        for i, sl in enumerate(pbar_win):
            if i % 200:
                pbar_win.set_description(f"Window {i}")
            yield Line(
                ds_id=ds_id,
                unit_id=unit_id,
                win_id=i,
                data=traj[features].iloc[sl].unstack().values,
                rul=traj["rul"].iloc[sl].iloc[-1],
            )


def feed_lmdb(
    output_lmdb: Path,
    iterator: Iterator[Line],
    bits: int,
    win_length: int,
    n_features: int,
) -> None:
    patterns: Dict[str, Callable[[Line], Union[bytes, np.ndarray]]] = {
        "{}": (
            lambda line: line.data.astype(np.float32)
            if bits == 32
            else line.data
        ),
        "ds_id_{}": (lambda line: "{}".format(line.ds_id).encode()),
        "unit_id_{}": (lambda line: "{}".format(line.unit_id).encode()),
        "win_id_{}": (lambda line: "{}".format(line.win_id).encode()),
        "rul_{}": (lambda line: "{}".format(line.rul).encode()),
    }
    return create_lmdb(
        filename=output_lmdb,
        iterator=iterator,
        patterns=patterns,
        aggs=[],
        win_length=win_length,
        n_features=n_features,
        bits=bits,
    )


def preprocess_lmdb(cfg: DictConfig):
    log.info("Preprocessing lmdb files ...")
    dm = NCMAPSSDataModule(
        cfg.paths.output_dir, batch_size=10000, all_dset=True
    )
    train_ds, val_ds = dm.random_split(cfg.dataset.val_ratio)
    scaler = StandardScalerTransform(
        train_dataloader=dm.train_dataloader(), sample_idx=-1
    )
    for lmdb, ds in zip(
        ["train_prep", "val_prep", "test_prep"],
        [train_ds, val_ds, dm.datasets["test"]],
    ):
        path = Path(f"{cfg.paths.output_dir}/lmdb/{lmdb}.lmdb")
        feed_lmdb(
            path,
            process_ds(ds, scaler),
            cfg.dataset.bits,
            cfg.dataset.win_length,
            len(get_features(cfg)),
        )
    for lmdb in ["train_prep", "val_prep", "test_prep"]:
        path = Path(f"{cfg.paths.output_dir}/lmdb/{lmdb}.lmdb")
        new_path = Path(
            f"{cfg.paths.output_dir}/lmdb/{path.stem.split('_')[0]}.lmdb"
        )
        shutil.copytree(
            path.as_posix(), new_path.as_posix(), dirs_exist_ok=True
        )
        shutil.rmtree(path.as_posix(), ignore_errors=True)


def process_ds(ds: Dataset, scaler: StandardScalerTransform) -> Iterator[Line]:
    for ds_id, unit_id, win_id, rul, sample in ds:
        yield Line(
            ds_id=ds_id,
            unit_id=unit_id,
            win_id=win_id,
            data=scaler(sample),
            rul=rul,
        )
