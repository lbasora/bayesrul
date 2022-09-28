from abc import ABC, abstractmethod
from typing import List, Union

from pytorch_lightning import LightningDataModule


class Inference(ABC):
    """Abstract class used to simplify benchmarking.
    Provided a LightningDataModule, initializes a model and offers methods
    to train, test and compute uncertainties on test set.
        data = LightningDataModule(...)
        inference = ...
        inference.fit(2)
        inference.test()
        inference.epistemic_aleatoric_uncertainty
    """

    name: str

    @abstractmethod
    def __init__(
        self,
        args,
        data: LightningDataModule,
        hyperparams=None,
        GPU=0,
        studying=False,
    ) -> None:
        ...

    @abstractmethod
    def _define_model(self):
        ...

    @abstractmethod
    def fit(
        self,
        epochs: int,
        monitor: Union[str, List[str]],
    ):
        ...

    @abstractmethod
    def test(
        self,
    ):
        ...

    @abstractmethod
    def epistemic_aleatoric_uncertainty(self, device=None):
        ...

    @property
    @abstractmethod
    def num_params(self) -> int:
        ...
