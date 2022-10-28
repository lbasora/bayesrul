import contextlib
import copy
from functools import partial

import pyro
import pyro.distributions as dist
import pytorch_lightning as pl
import torch
import tyxe
from attr import has
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from torch.functional import F
from tyxe.bnn import VariationalBNN

from ..results.metrics import rms_calibration_error, sharpness
from ..utils.miscellaneous import weights_init
from .guides.radial import AutoRadial


class BNN(pl.LightningModule):
    """
    Variational BNN Pytorch Lightning class, implements many subtleties needed
    to make tyxe work with Pytorch Lightning.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer,
        pretrain_epochs: bool,
        mc_samples_train: int,
        mc_samples_eval: int,
        dataset_size: int,
        fit_context: str,
        prior_loc: float,
        prior_scale: float,
        guide: str,
        q_scale: float,
    ):
        super().__init__()
        pyro.clear_param_store()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

    def define_bnn(self) -> None:
        if not self.hparams.pretrain_epochs == 0:
            self.net.apply(weights_init)

        prior_kwargs = {}  # {'hide_all': True}
        prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(
                    self.hparams.prior_loc,
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.tensor(
                    self.hparams.prior_scale,
                    dtype=torch.float32,
                    device=self.device,
                ),
            ),
            **prior_kwargs,
        )

        if self.hparams.fit_context == "lrt":
            self.fit_ctxt = tyxe.poutine.local_reparameterization
        elif self.hparams.fit_context == "flipout":
            self.fit_ctxt = tyxe.poutine.flipout
        else:
            self.fit_ctxt = contextlib.nullcontext

        likelihood = tyxe.likelihoods.HeteroskedasticGaussian(
            self.hparams.dataset_size,
            positive_scale=False,
        )

        guide_kwargs = {"init_scale": self.hparams.q_scale}
        if self.hparams.guide == "normal":
            guide_base = tyxe.guides.AutoNormal
        elif self.hparams.guide == "radial":
            guide_base = AutoRadial
            self.fit_ctxt = contextlib.nullcontext
        else:
            raise RuntimeError("Guide unknown. Choose from 'normal', 'radial'.")

        if self.hparams.pretrain_epochs > 0:
            guide_kwargs[
                "init_loc_fn"
            ] = tyxe.guides.PretrainedInitializer.from_net(self.net)
        guide = partial(guide_base, **guide_kwargs)

        self.bnn = VariationalBNN(
            copy.deepcopy(self.net.to(self.device)),
            prior,
            likelihood,
            guide,
        )

    def on_fit_start(self) -> None:
        self.define_bnn()
        param_store_to(self.device)
        self.configure_optimizers()

        self.loss = (
            TraceMeanField_ELBO(self.hparams.mc_samples_train)
            if self.hparams.guide != "radial"
            else Trace_ELBO(self.hparams.mc_samples_train)
        )

        self.svi = SVI(
            pyro.poutine.scale(
                self.bnn.model,
                scale=1.0
                / (
                    self.hparams.dataset_size
                    * self.net.win_length
                    * self.net.n_features
                ),
            ),
            pyro.poutine.scale(
                self.bnn.guide,
                scale=1.0
                / (
                    self.hparams.dataset_size
                    * self.net.win_length
                    * self.net.n_features
                ),
            ),
            self.hparams.optimizer,
            self.loss,
        )

    def training_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1]
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(
            self.bnn_no_obs, self.bnn.guide, self.hparams.optimizer, self.loss
        )

        # As we do not use PyTorch Optimizers, it is needed in order to Pytorch
        # Lightning to know that we are training a model, and initiate routines
        # like checkpointing etc.
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_ready()

        with self.fit_ctxt():
            elbo = self.svi.step(x, y.unsqueeze(-1))
            # Aggregate = False if num_prediction = 1, else nans in sd
            output = self.bnn.predict(
                x,
                num_predictions=self.hparams.mc_samples_train,
                aggregate=self.hparams.mc_samples_train > 1,
            ).squeeze()
            loc, scale = output[:, 0], output[:, 1]
            kl = self.svi_no_obs.evaluate_loss(x)
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.increment_completed()

        mse = F.mse_loss(y.squeeze(), loc.squeeze()).item()
        rmsce = rms_calibration_error(loc, scale, y.squeeze())
        sharp = sharpness(scale)
        self.log("mse/train", mse, on_step=False, on_epoch=True)
        self.log("elbo/train", elbo, on_step=False, on_epoch=True)
        self.log("kl/train", kl, on_step=False, on_epoch=True)
        self.log("likelihood/train", elbo - kl, on_step=False, on_epoch=True)
        self.log("rmsce/train", rmsce, on_step=False, on_epoch=True)
        self.log("sharp/train", sharp, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1]

        # To compute only the KL part of the loss (no obs = no likelihood)
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(
            self.bnn_no_obs, self.bnn.guide, self.hparams.optimizer, self.loss
        )

        elbo = self.svi.evaluate_loss(x, y.unsqueeze(-1))
        # Aggregate = False if num_prediction = 1, else nans in sd
        output = self.bnn.predict(
            x,
            num_predictions=self.hparams.mc_samples_eval,
            aggregate=self.hparams.mc_samples_eval > 1,
        ).squeeze()
        loc, scale = output[:, 0], output[:, 1]

        kl = self.svi_no_obs.evaluate_loss(x)

        mse = F.mse_loss(y.squeeze(), loc)
        rmsce = rms_calibration_error(loc, scale, y.squeeze())
        sharp = sharpness(scale)

        self.log("elbo/val", elbo)
        self.log("mse/val", mse)
        self.log("kl/val", kl)
        self.log("likelihood/val", elbo - kl)
        self.log("rmsce/val", rmsce)
        self.log("sharp/val", sharp)

    def on_test_start(self) -> None:
        self.test_preds = {
            "preds": [],
            "labels": [],
            "stds": [],
            "ep_vars": [],
            "al_vars": [],
        }
        self.define_bnn()
        param_store_to(self.device)

    def test_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1]
        output = self.bnn.predict(
            x,
            num_predictions=self.hparams.mc_samples_eval,
            aggregate=False,
        )
        loc, scale = output[:, :, 0], output[:, :, 1]

        ep_var = loc.var(0)
        al_var = (scale**2).mean(0)
        scale = al_var.add(ep_var).sqrt()
        loc = loc.mean(axis=0)

        nll = F.gaussian_nll_loss(loc, y, torch.square(scale))
        mse = F.mse_loss(y, loc)
        rmsce = rms_calibration_error(loc, scale, y)
        sharp = sharpness(scale)
        self.log("nll/test", nll)
        self.log("mse/test", mse)
        self.log("rmsce/test", rmsce)
        self.log("sharp/test", sharp)

        self.test_preds["preds"].append(loc.cpu().detach().numpy())
        self.test_preds["labels"].append(y.cpu().detach().numpy())
        self.test_preds["stds"].append(scale.cpu().detach().numpy())
        self.test_preds["ep_vars"].append(ep_var.cpu().detach().numpy())
        self.test_preds["al_vars"].append(al_var.cpu().detach().numpy())

    def configure_optimizers(self):
        return None

    def on_save_checkpoint(self, checkpoint):
        """Saving Pyro's param_store for the bnn's parameters"""
        checkpoint["param_store"] = pyro.get_param_store().get_state()

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        if not hasattr(self, "bnn"):
            checkpoint["state_dict"] = remove_dict_entry_startswith(
                checkpoint["state_dict"], "bnn"
            )


def param_store_to(device: str):
    ps = pyro.get_param_store().get_state()
    for k in ps["params"].keys():
        ps["params"][k] = ps["params"][k].to(device)
    pyro.get_param_store().set_state(ps)


def remove_dict_entry_startswith(dictionary, string):
    """Used to remove entries with 'bnn' in checkpoint state dict"""
    n = len(string)
    for key in dictionary:
        if string == key[:n]:
            dict2 = dictionary.copy()
            dict2.pop(key)
            dictionary = dict2
    return dictionary
