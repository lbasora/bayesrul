
from pathlib import Path

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from bayesrul.inference.inference import Inference
from bayesrul.lightning_wrappers.frequentist import DnnWrapper
from bayesrul.utils.miscellaneous import get_checkpoint, TBLogger, Dotdict, numel
from bayesrul.utils.plotting import ResultSaver



class DNN(Inference):
    """
    Standard frequentist neural networks
    """

    def __init__(
        self,
        args,
        data: pl.LightningDataModule,
        hyperparams = None,
        GPU = 1,
    ) -> None:    
        self.name = f"dnn_{args.model_name}_{args.archi}"
        assert isinstance(GPU, int), \
            f"GPU argument should be an int, not {type(GPU)}"
        assert isinstance(data, pl.LightningDataModule), \
            f"data argument should be a LightningDataModule, not {type(data)}"
        self.data = data
        self.GPU = GPU

        hyp = {
            'bias' : True,
            'lr' : 1e-3,
            'device' : torch.device(f"cuda:{self.GPU}"),
            'out_size' : 1,
        }
            
        if hyperparams is not None: # Overriding defaults with arguments
            for key in hyperparams.keys():
                hyp[key] = hyperparams[key]

        # Merge dicts and make attributes accessible by .
        self.args = Dotdict({**(args.__dict__), **hyp}) 

        self.base_log_dir = Path(args.out_path, "frequentist", args.model_name)

        self.logger = TBLogger(
            Path(self.base_log_dir),
            default_hp_metric=False,
        )


    def _define_model(self):
        self.checkpoint_file = get_checkpoint(self.base_log_dir, version=None)
        if self.checkpoint_file:
            self.dnn = DnnWrapper.load_from_checkpoint(self.checkpoint_file,
                map_location=self.args.device)
        else:
            self.dnn = DnnWrapper(
                self.data.win_length, 
                self.data.n_features, 
                **self.args,
            )


    def fit(self, epochs):
        if not hasattr(self, 'dnn'):
            self._define_model()

        self.monitor = f"{self.dnn.loss}/val"
        earlystopping_callback = EarlyStopping(monitor=self.monitor, patience=50)

        self.trainer = pl.Trainer(
            default_root_dir=self.base_log_dir,
            gpus=[1],
            max_epochs=epochs,
            log_every_n_steps=2,
            logger=self.logger,
            callbacks=[
                earlystopping_callback,
            ],
        )

        self.trainer.fit(self.dnn, self.data, ckpt_path=self.checkpoint_file)



    def test(self):
        if not hasattr(self, 'dnn'):
            self._define_model()

        tester = pl.Trainer(
            gpus=[0], 
            log_every_n_steps=10, 
            logger=self.logger, 
            max_epochs=-1
        ) # Silence warning
        
        tester.test(self.dnn, self.data, verbose=False)
        
        self.results = ResultSaver(self.base_log_dir)
        self.results.save(self.dnn.test_preds)


    def epistemic_aleatoric_uncertainty(self):
        self.dnn.test_preds
        ...

    def num_params(self) -> int:
        return numel(self.dnn.net)