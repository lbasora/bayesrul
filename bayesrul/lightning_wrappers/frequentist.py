# type: ignore
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.functional import F

import numpy as np
from bayesrul.models.conv import Conv
from bayesrul.models.inception import BigCeption, InceptionModel
from bayesrul.models.linear import Linear
from bayesrul.utils.metrics import rms_calibration_error, sharpness
from bayesrul.utils.miscellaneous import enable_dropout, weights_init


class DnnWrapper(pl.LightningModule):
    """
    Pytorch Lightning frequentist models wrapper
    This class is used by frequentist models, MC_Dropout and Heteroscedastic NNs

    It implements various functions for Pytorch Lightning to manage train, test,
    validation, logging...
    """

    def __init__(
        self,
        win_length,
        n_features,
        bias=True,
        archi="inception",
        out_size=1,
        lr=1e-3,
        weight_decay=1e-3,
        loss="mse",
        activation="relu",
        dropout=0,
        device=torch.device("cuda:0"),
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if archi == "linear":
            self.net = Linear(
                win_length,
                n_features,
                activation=activation,
                dropout=dropout,
                bias=bias,
                out_size=out_size,
            ).to(device)
        elif archi == "conv":
            self.net = Conv(
                win_length,
                n_features,
                activation=activation,
                dropout=dropout,
                bias=bias,
                out_size=out_size,
            ).to(device)
        elif archi == "inception":
            self.net = InceptionModel(
                win_length,
                n_features,
                out_size=out_size,
                dropout=dropout,
                activation=activation,
                bias=bias,
            ).to(device)
        elif archi == "bigception":
            self.net = BigCeption(
                win_length,
                n_features,
                activation=activation,
                dropout=dropout,
                out_size=out_size,
                bias=bias,
            ).to(device)
        else:
            raise RuntimeError(f"Model architecture {archi} not implemented")

        if out_size == 2:
            self.loss = "gaussian_nll"
            self.criterion = F.gaussian_nll_loss
        elif (loss == "mse") or (loss == "MSE"):
            self.criterion = F.mse_loss
            self.loss = "mse"
        elif (loss == "l1") or (loss == "L1"):
            self.criterion = F.l1_loss
            self.loss = "l1"
        else:
            raise RuntimeError(
                f"Loss {loss} not supported or out_size {out_size}"
                "not adapted to loss. Choose from ['mse', 'l1'] for out_size=1"
            )

        self.out_size = out_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_preds = {"preds": [], "labels": []}
        if dropout > 0:
            self.test_preds["stds"] = []
        self.dropout = dropout
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def get_device(self):
        return next(self.net.parameters()).device

    def to_device(self, device: torch.device):
        # print(f"Device before to {self.get_device()}")
        self.net.to(device)
        # print(f"Device After to {self.get_device()}")

    def _compute_loss(self, batch, phase, return_pred=False):
        (x, y) = batch
        output = self.net(x)
        if output.shape[1] == 2:
            y_hat = output[:, 0]
            scale = output[:, 1]
        else:
            y_hat = output.squeeze()

        if self.loss == "gaussian_nll":
            loss = self.criterion(y_hat, y, torch.square(scale))
        else:
            loss = self.criterion(y_hat, y)

        self.log(f"{self.loss}/{phase}", loss)
        if return_pred:
            if output.shape[1] == 2:
                return loss, y_hat, scale
            else:
                return loss, y_hat
        else:
            return loss

    def training_step(self, batch, batch_idx):
        if self.out_size == 2:
            loss, loc, scale = self._compute_loss(
                batch, "train", return_pred=True
            )
            mse = F.mse_loss(loc, batch[1])
            rmsce = rms_calibration_error(loc, scale, batch[1])
            sharp = sharpness(scale)
            self.log("mse/train", mse)
            self.log("rmsce/train", rmsce)
            self.log("sharp/train", sharp)
        else:
            loss = self._compute_loss(batch, "train", return_pred=False)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.dropout > 0:  # MC-Dropout
            enable_dropout(self.net)
            preds = []
            losses = []
            scales = []
            for i in range(10):
                if self.out_size == 2:
                    loss, loc, scale = self._compute_loss(
                        batch, "val", return_pred=True
                    )
                    scales.append(scale)
                else:
                    loss, loc = self._compute_loss(
                        batch, "val", return_pred=True
                    )
                preds.append(loc)
                losses.append(loss)
            loss = torch.stack(losses).mean()
            preds = torch.stack(preds)
            if self.out_size == 2:
                scales = torch.stack(scales)
                scale = scales.mean(axis=0)
            else:
                scale = preds.std(axis=0)
            preds = preds.mean(axis=0)
            self.log(f"{self.loss}/val", loss)
            return {
                "loss": loss,
                "label": batch[1],
                "pred": preds,
                "std": scale,
            }
        elif self.out_size == 2:  # Not MC Dropout
            loss, pred, std = self._compute_loss(batch, "val", return_pred=True)
            return {"loss": loss, "label": batch[1], "pred": pred, "std": std}
        else:
            loss = self._compute_loss(batch, "val", return_pred=False)
            return {"loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        if (self.dropout > 0) | (self.out_size == 2):

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
        if self.dropout > 0:
            enable_dropout(self.net)
            preds = []
            losses = []
            scales = []
            for i in range(100):
                if self.out_size == 2:
                    loss, loc, scale = self._compute_loss(
                        batch, "test", return_pred=True
                    )
                    scales.append(scale)
                else:
                    loss, loc = self._compute_loss(
                        batch, "test", return_pred=True
                    )
                preds.append(loc)
                losses.append(loss)

            loss = torch.stack(losses).mean()
            preds = torch.stack(preds)
            if self.out_size == 2:
                scales = torch.stack(scales)
                scale = scales.mean(axis=0)
            else:
                scale = preds.std(axis=0)
            preds = preds.mean(axis=0)

            return {
                "loss": loss,
                "label": batch[1],
                "pred": preds,
                "std": scale,
            }
        elif (self.out_size == 2) & (self.dropout == 0):
            loss, pred, std = self._compute_loss(
                batch, "test", return_pred=True
            )
            return {"loss": loss, "label": batch[1], "pred": pred, "std": std}
        else:
            loss, pred = self._compute_loss(batch, "test", return_pred=True)
            return {"loss": loss, "label": batch[1], "pred": pred}

    def test_epoch_end(self, outputs):
        for i, output in enumerate(outputs):
            if i == 0:
                preds = output["pred"].detach()
                labels = output["label"].detach()
                if (self.dropout > 0) | (self.loss == "gaussian_nll"):
                    stds = output["std"].detach()
            else:
                preds = torch.cat([preds, output["pred"].detach()])
                labels = torch.cat([labels, output["label"].detach()])
                if (self.dropout > 0) | (self.loss == "gaussian_nll"):
                    stds = torch.cat([stds, output["std"].detach()])

        self.test_preds["preds"] = preds.cpu().numpy()
        self.test_preds["labels"] = labels.cpu().numpy()

        mse = F.mse_loss(preds, labels)
        self.log("mse/test", mse)

        if (self.dropout > 0) | (self.loss == "gaussian_nll"):
            self.test_preds["stds"] = stds.cpu().numpy()

            rmsce = rms_calibration_error(preds, stds, labels)
            sharp = sharpness(stds)
            self.log("rmsce/test", rmsce)
            self.log("sharp/test", sharp)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args"""
        parser = parent_parser.add_argument_group("DnnWrapper")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser


class DnnPretrainWrapper(pl.LightningModule):
    """
    Class used by BNNs to pretrain their weights. This class is instantiated,
    trained for X epochs and then it stores its weights in the log directory.
    VIBnnWrapper then loads the weights and starts the Bayesian training
    """

    def __init__(
        self,
        win_length,
        n_features,
        bias=True,
        archi="linear",
        out_size=2,
        activation="relu",
        lr=1e-3,
        weight_decay=1e-3,
        loss="mse",
    ):
        super().__init__()
        self.save_hyperparameters()
        if archi == "linear":
            self.net = Linear(
                win_length,
                n_features,
                activation=activation,
                bias=bias,
                out_size=out_size,
            )
        elif archi == "conv":
            self.net = Conv(
                win_length,
                n_features,
                activation=activation,
                bias=bias,
                out_size=out_size,
            )
        elif archi == "inception":
            self.net = InceptionModel(
                win_length,
                n_features,
                out_size=out_size,
                activation=activation,
                bias=bias,
            )
        elif archi == "bigception":
            self.net = BigCeption(
                win_length,
                n_features,
                out_size=out_size,
                activation=activation,
                bias=bias,
            )
        else:
            raise RuntimeError(f"Model architecture {archi} not implemented")

        if (loss == "mse") or (loss == "MSE"):
            self.criterion = F.mse_loss
            self.loss = "mse"
        elif (loss == "l1") or (loss == "L1"):
            self.criterion = F.l1_loss
            self.loss = "l1"
        else:
            raise RuntimeError(
                f"Loss {loss} not supported. Choose from" " ['mse', 'l1']"
            )

        self.lr = lr
        self.weight_decay = weight_decay
        self.net.apply(weights_init)

    def forward(self, x):
        return self.net(x)

    def _compute_loss(self, batch, phase, return_pred=False):
        (x, y) = batch
        y = y.view(-1, 1).to(torch.float32)
        output = self.net(x)
        try:
            y_hat = output[:, 0]
            sd = output[:, 1]
        except:
            y_hat = output
        # print(f"Output shape {output.shape}, then y_hat {y_hat.shape} then y {y.shape}")
        loss = self.criterion(y_hat, y.squeeze())
        if return_pred:
            return loss, y_hat
        else:
            return loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args"""
        parser = parent_parser.add_argument_group("DnnPretrainWrapper")
        parser.add_argument("--net", type=str, default="linear")
        return parent_parser
