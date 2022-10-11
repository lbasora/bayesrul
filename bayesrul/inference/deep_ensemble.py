from pathlib import Path

import pytorch_lightning as pl
import torch

import numpy as np
from bayesrul.inference.dnn import HeteroscedasticDNN
from bayesrul.inference.inference import Inference
from bayesrul.utils.post_process import ResultSaver


class DeepEnsemble(Inference):
    """
    Deep Ensemble neural networks inference class
    """

    def __init__(
        self,
        args,
        data: pl.LightningDataModule,
        n_models: int,
        hyperparams=None,
        GPU=0,
        studying=False,
    ) -> None:

        assert isinstance(
            GPU, int
        ), f"GPU argument should be an int, not {type(GPU)}"
        assert isinstance(
            n_models, int
        ), f"n_models argument should be an int, not {type(n_models)}"
        assert isinstance(
            data, pl.LightningDataModule
        ), f"data argument should be a LightningDataModule, not {type(data)}"
        self.data = data
        self.GPU = GPU
        self.n_models = n_models

        hyp = {
            "bias": True,
            "lr": 1e-3,
            "device": torch.device(f"cuda:{self.GPU}"),
            "dropout": 0,
            "out_size": 2,
        }

        if hyperparams is not None:  # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]
        self.hyp = hyp

        self.args = args

        directory = "studies" if studying else "frequentist"
        self.base_log_dir = Path(args.out_path, directory, args.model_name)

    def _define_model(self):
        self.models = [
            HeteroscedasticDNN(
                self.args, self.data, self.hyp, GPU=self.GPU, version=i
            )
            for i in range(self.n_models)
        ]

    def fit(self, epochs, monitor=None, early_stop=0):
        if not hasattr(self, "models"):
            self._define_model()

        metrics = [
            model.fit(epochs, monitor, early_stop) for model in self.models
        ]
        return np.asarray(metrics).mean()

    def test(self):
        if not hasattr(self, "models"):
            self._define_model()

        test_preds = dict()
        test_preds["labels"] = None
        test_preds["preds"] = []
        test_preds["stds"] = []
        for model in self.models:
            model_test_preds = model.test()
            if test_preds["labels"] is None:
                test_preds["labels"] = model_test_preds["labels"]
            test_preds["preds"].append(model_test_preds["preds"])
            test_preds["stds"].append(model_test_preds["stds"])

        loc = np.stack(test_preds["preds"])
        test_preds["preds"] = loc.mean(0)
        # Gaussian mixture formula : var = (scale**2+loc**2).mean(0) - loc.mean(0)**2)
        loc, scale = test_preds["preds"], np.stack(test_preds["stds"])
        test_preds["stds"] = np.sqrt(
            (scale**2 + loc**2).mean(0) - test_preds["preds"] ** 2
        )
        self.results = ResultSaver(self.base_log_dir)
        self.results.save(test_preds)

    def epistemic_aleatoric_uncertainty(self, device=None):
        raise NotImplementedError(
            "Deep Ensembles can't model epistemic uncertainties."
        )

    def num_params(self) -> int:
        if not hasattr(self, "models"):
            self._define_model()

        return np.sum([model.num_params() for model in self.models])
