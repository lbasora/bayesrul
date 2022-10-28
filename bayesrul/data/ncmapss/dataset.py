from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from ..lmdb_utils import LmdbDataset


class NCMAPSSLmdbDataset(LmdbDataset):
    """Returns features X + rul Y for training purposes"""

    def __getitem__(self, i: int):
        sample = super().__getitem__(i)
        rul = self.dtype(super().get(f"rul_{i}", numpy=False))
        return sample.copy(), rul


class NCMAPSSLmdbDatasetAll(NCMAPSSLmdbDataset):
    """Returns features X + other data + rul Y"""

    def __getitem__(self, i: int):
        sample, rul = super().__getitem__(i)
        ds_id = int(super().get(f"ds_id_{i}", numpy=False))
        unit_id = int(float(super().get(f"unit_id_{i}", numpy=False)))
        win_id = int(super().get(f"win_id_{i}", numpy=False))
        return ds_id, unit_id, win_id, rul, sample


class NCMAPSSDataModule(pl.LightningDataModule):
    """
    Instantiates LMDB reader for train, test and val, and constructs Pytorch
    Lightning loaders. This is the way to access generated LMDBs
    """

    def __init__(
        self,
        data_path,
        batch_size,
        test_batch_size=10000,
        all_dset=False,
        num_workers=12,
        pin_memory=True,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.all_dset = all_dset

        ds_list = ["train", "test"]
        if Path(f"{self.data_path}/lmdb/val.lmdb").exists():
            ds_list.append("val")
        self.datasets = dict(
            (
                name,
                NCMAPSSLmdbDatasetAll(
                    f"{self.data_path}/lmdb/{name}.lmdb",
                    "{}",
                )
                if self.all_dset
                else NCMAPSSLmdbDataset(
                    f"{self.data_path}/lmdb/{name}.lmdb",
                    "{}",
                ),
            )
            for name in ds_list
        )
        self.win_length, self.n_features = (
            int(self.datasets["train"].get("win_length", numpy=False)),
            self.datasets["train"].n_features,
        )

    @property
    def train_size(self) -> int:
        return len(self.datasets["train"])

    def random_split(self, val_ratio: float) -> tuple:
        val_len = int(val_ratio * len(self.datasets["train"]))
        train_len = len(self.datasets["train"]) - val_len
        return random_split(
            self.datasets["train"],
            [train_len, val_len],
            generator=torch.Generator().manual_seed(0),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.datasets["test"],
            batch_size=self.test_batch_size,
            shuffle=False,  # Important. do NOT shuffle or results will be false
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
