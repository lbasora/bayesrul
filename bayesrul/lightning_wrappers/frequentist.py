import pytorch_lightning as pl
import torch
from torch.functional import F

from bayesrul.utils.metrics import rms_calibration_error, sharpness
from bayesrul.utils.miscellaneous import enable_dropout, weights_init


class HNN(pl.LightningModule):
    """
    Pytorch Lightning frequentist models wrapper
    This class is used by frequentist models, MC_Dropout and Heteroscedastic NNs

    It implements various functions for Pytorch Lightning to manage train, test,
    validation, logging...
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mc_samples_train: int,
        mc_samples_eval: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=["net"])
        self.net = net
        self.test_preds = {
            "preds": [],
            "labels": [],
            "stds": [],
        }
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def get_device(self):
        return next(self.net.parameters()).device

    def to_device(self, device: torch.device):
        # print(f"Device before to {self.get_device()}")
        self.net.to(device)
        # print(f"Device After to {self.get_device()}")

    def step(self, batch, phase):
        (x, y) = batch
        output = self.net(x)
        loc = output[:, 0]
        scale = output[:, 1]
        loss = F.gaussian_nll_loss(loc, y, torch.square(scale))
        self.log(f"nll/{phase}", loss, on_step=False, on_epoch=True)
        return loss, loc, scale

    def mc_sampling(self, batch, mc_samples: int, phase: str):
        enable_dropout(self.net)
        losses = []
        locs = []
        scales = []
        for _ in range(mc_samples):
            loss, loc, scale = self.step(batch, phase)
            losses.append(loss)
            locs.append(loc)
            scales.append(scale)
        loss = torch.stack(losses).mean()
        locs = torch.stack(locs)
        scales = torch.stack(scales)
        scale = scales.pow(2).mean(0).add(locs.var(0)).sqrt()
        loc = locs.mean(axis=0)
        return loss, loc, scale

    def training_step(self, batch, batch_idx):
        if self.net.dropout > 0:  # MC-Dropout
            loss, loc, scale = self.mc_sampling(
                batch, self.hparams.mc_samples_train, phase="train"
            )
        else:
            loss, loc, scale = self.step(batch, "train")
        mse = F.mse_loss(loc, batch[1])
        rmsce = rms_calibration_error(loc, scale, batch[1])
        sharp = sharpness(scale)
        self.log("mse/train", mse, on_step=False, on_epoch=True)
        self.log("rmsce/train", rmsce, on_step=False, on_epoch=True)
        self.log("sharp/train", sharp, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        phase = "val"
        if self.net.dropout > 0:
            loss, loc, scale = self.mc_sampling(
                batch, self.hparams.mc_samples_eval, phase=phase
            )
        else:
            loss, loc, scale = self.step(batch, phase)
        return {"loss": loss, "label": batch[1], "pred": loc, "std": scale}

    def validation_epoch_end(self, outputs) -> None:
        for i, output in enumerate(outputs):
            if i == 0:
                preds = output["pred"].detach()
                labels = output["label"].detach()
                stds = output["std"].detach()
            else:
                preds = torch.cat([preds, output["pred"].detach()])
                labels = torch.cat([labels, output["label"].detach()])
                stds = torch.cat([stds, output["std"].detach()])

        mse = F.mse_loss(preds, labels)
        rmsce = rms_calibration_error(preds, stds, labels)
        sharp = sharpness(stds)
        self.log("mse/val", mse)
        self.log("rmsce/val", rmsce)
        self.log("sharp/val", sharp)

    def test_step(self, batch, batch_idx):
        phase = "test"
        if self.net.dropout > 0:
            loss, loc, scale = self.mc_sampling(
                batch, self.hparams.mc_samples_eval, phase=phase
            )
        else:
            loss, loc, scale = self.step(batch, phase)
        return {"loss": loss, "label": batch[1], "pred": loc, "std": scale}

    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            if i == 0:
                preds = output["pred"].detach()
                labels = output["label"].detach()
                stds = output["std"].detach()
            else:
                preds = torch.cat([preds, output["pred"].detach()])
                labels = torch.cat([labels, output["label"].detach()])
                stds = torch.cat([stds, output["std"].detach()])
        self.test_preds["preds"] = preds.cpu().numpy()
        self.test_preds["labels"] = labels.cpu().numpy()
        self.test_preds["stds"] = stds.cpu().numpy()
        self.log("mse/test", F.mse_loss(preds, labels))
        self.log(
            "nll/test", F.gaussian_nll_loss(preds, labels, torch.square(stds))
        )
        self.log("rmsce/test", rms_calibration_error(preds, stds, labels))
        self.log("sharp/test", sharpness(stds))

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())


class NN(pl.LightningModule):
    """
    Class used by BNNs to pretrain their weights. This class is instantiated,
    trained for X epochs and then it stores its weights in the log directory.
    VIBnnWrapper then loads the weights and starts the Bayesian training
    """

    def __init__(self, net: torch.nn.Module, optimizer: torch.optim.Optimizer):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def step(self, batch):
        (x, y) = batch
        y = y.view(-1, 1).to(torch.float32)
        output = self.net(x)
        y_hat = output[:, 0]
        return F.mse_loss(y_hat, y.squeeze())

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        mse = self.step(batch)
        self.log("mse/val", mse, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())
