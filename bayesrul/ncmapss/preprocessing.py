import logging
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Union

import h5py
from torch.utils.data import Dataset
from tqdm import tqdm

import numpy as np
import pandas as pd
from bayesrul.ncmapss.dataset import NCMAPSSDataModule

from ..utils.lmdb_utils import StandardScalerTransform, create_lmdb, make_slice

ncmapss_files = [
    "N-CMAPSS_DS01-005",
    "N-CMAPSS_DS02-006",
    "N-CMAPSS_DS03-012",
    "N-CMAPSS_DS04",
    "N-CMAPSS_DS05",
    "N-CMAPSS_DS06",
    "N-CMAPSS_DS07",
    "N-CMAPSS_DS08a-009",
    "N-CMAPSS_DS08c-008",
    "N-CMAPSS_DS08d-010",
]

ncmapss_data = ["X_s", "X_v", "T", "A"]

ncmapss_datanames = {
    "W": ["alt", "Mach", "TRA", "T2"],
    "X_s": [
        "T24",
        "T30",
        "T48",
        "T50",
        "P15",
        "P2",
        "P21",
        "P24",
        "Ps30",
        "P40",
        "P50",
        "Nf",
        "Nc",
        "Wf",
    ],
    "X_v": [
        "T40",
        "P30",
        "P45",
        "W21",
        "W22",
        "W25",
        "W31",
        "W32",
        "W48",
        "W50",
        "SmFan",
        "SmLPC",
        "SmHPC",
        "phi",
    ],
    "T": [
        "fan_eff_mod",
        "fan_flow_mod",
        "LPC_eff_mod",
        "LPC_flow_mod",
        "HPC_eff_mod",
        "HPC_flow_mod",
        "HPT_eff_mod",
        "HPT_flow_mod",
        "LPT_eff_mod",
        "LPT_flow_mod",
    ],
    "A": [],  # ['Fc', 'unit', 'cycle', 'hs'] removed because not judged relevant
    "Y": ["rul"],
}


def generate_lmdb(args) -> None:
    """Parquet files to lmdb files"""
    lmdb_dir = Path(f"{args.out_path}/lmdb")
    lmdb_dir.mkdir(exist_ok=True)
    lmdb_dir_files = [x for x in lmdb_dir.iterdir()]
    if len(lmdb_dir_files) > 0:
        warnings.warn(
            f"{lmdb_dir} is not empty. Generation will not overwrite"
            " the previously generated .lmdb files. It will append data."
        )
    features = []
    features.extend(ncmapss_datanames["W"])  # We treat W as input data
    for key in ncmapss_datanames:
        if (key == "Y") or (key not in args.subdata):
            continue
        else:
            features.extend(ncmapss_datanames[key])
    for ds in ["train", "test"]:
        filelist = list(
            Path(args.data_path, filename) for filename in args.files
        )
        logging.info(
            f"Generating {ds} lmdb with {[x.as_posix() for x in filelist]} files..."
        )
        if filelist is not None:
            iterator = process_files(
                filelist,
                "dev" if ds == "train" else "test",
                args.bits,
                args.win_length,
                args.win_step,
                features,
                args.subdata,
                args.skip_obs,
            )
            feed_lmdb(Path(f"{lmdb_dir}/{ds}.lmdb"), iterator, args)


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
    features: List[str],
    subdata: List[str],
    skip_obs: Optional[int] = None,
) -> Iterator[Line]:

    for filename in tqdm(filelist):
        df = load_data_from_file(
            filename,
            dev_test,
            np.float32 if bits == 32 else np.float64,
            vars=subdata,
        )
        if "A" in subdata:
            df = linear_piece_wise_RUL(df.copy())
        if skip_obs is not None:
            df = df[::skip_obs]
        ds_id = int(str(filename.stem).split("_")[-1].split("-")[0][3])
        yield from process_dataframe(df, ds_id, win_length, win_step, features)
        del df


def load_data_from_file(
    filepath, dev_test_suffix, dtype=np.float64, vars=["X_s", "X_v", "T", "A"]
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
        [x in ["X_s", "X_v", "T", "A"] for x in vars]
    ), "Wrong vars provided, choose a subset of ['X_s', 'X_v', 'T', 'A']"
    assert filepath.name in ncmapss_files, "Incorrect file name {}".format(
        filepath
    )

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


def feed_lmdb(output_lmdb: Path, iterator, args) -> None:
    patterns: Dict[str, Callable[[Line], Union[bytes, np.ndarray]]] = {
        "{}": (
            lambda line: line.data.astype(np.float32)
            if args.bits == 32
            else line.data
        ),
        "ds_id_{}": (lambda line: "{}".format(line.ds_id).encode()),
        "unit_id_{}": (lambda line: "{}".format(line.unit_id).encode()),
        "win_id_{}": (lambda line: "{}".format(line.win_id).encode()),
        "rul_{}": (lambda line: "{}".format(line.rul).encode()),
    }
    features = []
    features.extend(ncmapss_datanames["W"])  # We treat W as input data
    for key in ncmapss_datanames:
        if (key == "Y") or (key not in args.subdata):
            continue
        else:
            features.extend(ncmapss_datanames[key])

    return create_lmdb(
        filename=output_lmdb,
        iterator=iterator,
        patterns=patterns,
        aggs=[],
        win_length=args.win_length,
        n_features=len(features),
        bits=args.bits,
    )


def preprocess_lmdb(args):
    logging.info("Preprocessing lmdb files ...")
    dm = NCMAPSSDataModule(args.out_path, batch_size=10000, all_dset=True)
    train_ds, val_ds = dm.random_split(args.val_ratio)
    scaler = StandardScalerTransform(
        train_dataloader=dm.train_dataloader(), sample_idx=-1
    )
    for lmdb, ds in zip(
        ["train_prep", "val_prep", "test_prep"],
        [train_ds, val_ds, dm.datasets["test"]],
    ):
        path = Path(f"{args.out_path}/lmdb/{lmdb}.lmdb")
        feed_lmdb(
            path,
            process_ds(ds, scaler),
            args,
        )
    for lmdb in ["train_prep", "val_prep", "test_prep"]:
        path = Path(f"{args.out_path}/lmdb/{lmdb}.lmdb")
        new_path = Path(f"{args.out_path}/lmdb/{path.stem.split('_')[0]}.lmdb")
        shutil.copytree(
            path.as_posix(), new_path.as_posix(), dirs_exist_ok=True
        )
        shutil.rmtree(path.as_posix(), ignore_errors=True)


def process_ds(ds: Dataset, scaler: StandardScalerTransform) -> Line:
    for ds_id, unit_id, win_id, rul, sample in ds:
        yield Line(
            ds_id=ds_id,
            unit_id=unit_id,
            win_id=win_id,
            data=scaler(sample),
            rul=rul,
        )
