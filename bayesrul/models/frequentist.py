import pytorch_lightning as pl
import torch
from torch.functional import F

from ..results.metrics import rms_calibration_error, sharpness
from ..utils.miscellaneous import enable_dropout, weights_init


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
        mc_samples: int,
        p_dropout: int,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def get_device(self):
        return next(self.net.parameters()).device

    def to_device(self, device: torch.device):
        self.net.to(device)

    def step(self, batch, phase):
        (x, y) = batch
        output = self.net(x)
        loc = output[:, 0]
        scale = output[:, 1]
        if phase == "predict":
            return loc, scale
        loss = F.gaussian_nll_loss(loc, y, torch.square(scale))
        self.log(f"nll/{phase}", loss, on_step=False, on_epoch=True)
        return loss, loc, scale

    def training_step(self, batch, batch_idx):
        loss, loc, scale = self.step(batch, "train")
        mse = F.mse_loss(loc, batch[1])
        rmsce = rms_calibration_error(loc, scale, batch[1])
        sharp = sharpness(scale)
        self.log("mse/train", mse, on_step=False, on_epoch=True)
        self.log("rmsce/train", rmsce, on_step=False, on_epoch=True)
        self.log("sharp/train", sharp, on_step=False, on_epoch=True)
        return loss

    def mc_sampling(self, batch, mc_samples: int, phase: str, agg: bool = True):
        losses = []
        locs = []
        scales = []
        for _ in range(mc_samples):
            if phase == "predict":
                loc, scale = self.step(batch, phase)
            else:
                loss, loc, scale = self.step(batch, phase)
                losses.append(loss)
            locs.append(loc)
            scales.append(scale)
        locs = torch.stack(locs)
        scales = torch.stack(scales)
        if phase == "predict":
            return locs, scales
        loss = torch.stack(losses).mean(0)
        if agg:
            scale = scales.pow(2).mean(0).add(locs.var(0)).sqrt()
            loc = locs.mean(0)
            return loss, loc, scale
        return loss, locs, scales

    def validation_step(self, batch, batch_idx):
        phase = "val"
        if self.net.dropout > 0:
            enable_dropout(self.net)
            loss, loc, scale = self.mc_sampling(
                batch, self.hparams.mc_samples, phase=phase
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
        y = batch[1]
        phase = "test"
        if self.net.dropout > 0:
            enable_dropout(self.net)
            loss, locs, scales = self.mc_sampling(
                batch, self.hparams.mc_samples, phase=phase, agg=False
            )
            ep_var = locs.var(0)
            al_var = (scales**2).mean(0)
            scale = al_var.add(ep_var).sqrt()
            loc = locs.mean(axis=0)
        else:
            loss, loc, scale = self.step(batch, phase)

        self.log("nll/test", loss)
        self.log("mse/test", F.mse_loss(loc, y))
        self.log("rmsce/test", rms_calibration_error(loc, scale, y))
        self.log("sharp/test", sharpness(scale))

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred = dict()
        pred["labels"] = batch[1].cpu().numpy()
        phase = "predict"
        if self.net.dropout > 0:
            enable_dropout(self.net)
            locs, scales = self.mc_sampling(
                batch, self.hparams.mc_samples, phase=phase, agg=False
            )
            ep_var = locs.var(0)
            al_var = (scales**2).mean(0)
            scale = al_var.add(ep_var).sqrt()
            loc = locs.mean(axis=0)
            pred["ep_vars"] = ep_var.cpu().numpy()
            pred["al_vars"] = al_var.cpu().numpy()
        else:
            loc, scale = self.step(batch, phase)
        pred["preds"] = loc.cpu().numpy()
        pred["stds"] = scale.cpu().numpy()
        return pred

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
