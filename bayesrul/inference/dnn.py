# type: ignore
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from bayesrul.inference.inference import Inference
from bayesrul.lightning_wrappers.frequentist import DnnWrapper
from bayesrul.utils.miscellaneous import (
    Dotdict,
    get_checkpoint,
    numel,
)
from bayesrul.utils.post_process import ResultSaver


class HeteroscedasticDNN(Inference):
    """
    Heteroscedastic frequentist neural networks
    """

    def __init__(
        self,
        args,
        data: pl.LightningDataModule,
        hyperparams=None,
        GPU=1,
        studying=False,
    ) -> None:
        assert isinstance(
            GPU, int
        ), f"GPU argument should be an int, not {type(GPU)}"
        assert isinstance(
            data, pl.LightningDataModule
        ), f"data argument should be a LightningDataModule, not {type(data)}"
        self.data = data
        self.GPU = GPU

        hyp = {
            "bias": True,
            "lr": 1e-3,
            "device": torch.device(f"cuda:{self.GPU}"),
            "weight_decay": 0,
            "dropout": 0,  # Do not change this, or will be considered MC-Dropout
        }

        if hyperparams is not None:  # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        # Merge dicts and make attributes accessible by .
        self.args = Dotdict({**(args.__dict__), **hyp})
        self.args.out_size = 2

        directory = "studies" if studying else "frequentist"
        self.base_log_dir = Path(args.out_path, directory, args.model_name)

    def _define_model(self):
        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        if self.checkpoint_file:
            self.dnn = DnnWrapper.load_from_checkpoint(
                self.checkpoint_file, map_location=self.args.device
            )
        else:
            self.dnn = DnnWrapper(
                self.data.win_length,
                self.data.n_features,
                **self.args,
            )

    def fit(self, epochs, monitor, early_stop=0):
        if not hasattr(self, "dnn"):
            self._define_model()

        self.trainer = pl.Trainer(
            default_root_dir=self.base_log_dir,
            accelerator="gpu",
            devices=[self.GPU],
            max_epochs=epochs,
            log_every_n_steps=20,
            callbacks=[
                ModelCheckpoint(monitor=monitor),
                EarlyStopping(monitor=monitor, patience=early_stop),
            ]
            if early_stop
            else None,
        )

        self.trainer.fit(self.dnn, self.data, ckpt_path=self.checkpoint_file)

        return self.trainer.callback_metrics[monitor]

    def test(self):
        if not hasattr(self, "dnn"):
            self._define_model()

        tester = pl.Trainer(
            accelerator="gpu",
            devices=[self.GPU],
            log_every_n_steps=100,
            max_epochs=-1,
        )  # Silence warning

        tester.test(self.dnn, self.data, verbose=False)

        self.results = ResultSaver(self.base_log_dir)
        self.results.save(self.dnn.test_preds)

    def epistemic_aleatoric_uncertainty(self):
        raise RuntimeError("Heteroscedastic NN can not separate uncertainties.")

    def num_params(self) -> int:
        return numel(self.dnn.net)
