import contextlib
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn as nn
from bayesrul.utils.miscellaneous import weights_init
from bayesrul.models.inception import InceptionModel, BigCeption
from bayesrul.models.linear import Linear
from bayesrul.models.conv import Conv
from tyxe.bnn import VariationalBNN
from bayesrul.utils.radial import AutoRadial
from torch.functional import F


from torch.autograd import detect_anomaly

import tyxe
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, MCMC, Trace_ELBO, JitTrace_ELBO
from pyro.infer import TraceMeanField_ELBO, JitTraceMeanField_ELBO

def remove_dict_entry_startswith(dictionary, string):
    """Used to remove entries with 'bnn' in checkpoint state dict"""
    n = len(string)
    for key in dictionary:
        if string == key[:n]:
            dict2 = dictionary.copy()
            dict2.pop(key)
            dictionary = dict2
    return dictionary


class BnnWrapper(pl.LightningModule):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        bias=True,
        archi='inception',
        activation='relu',
        optimizer='nadam',
        lr=1e-3,
        device=torch.device('cuda:0'),
        **kwargs
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.lr = lr

        if archi == "linear":
            self.net = Linear(win_length, n_features, activation=activation,
                bias=bias, typ='regression').to(device=device)
        elif archi == "conv":
            self.net = Conv(win_length, n_features, activation=activation, 
                bias=bias, typ='regression').to(device=device)
        elif archi == "inception":
            self.net = InceptionModel(activation=activation, 
                bias=bias).to(device=device)
        elif archi == "bigception":
            self.net = BigCeption(n_features, activation=activation, 
                bias=bias).to(device=device)
        else:
            raise ValueError(f"Model architecture {archi} not implemented")

        with torch.no_grad():
            self.net.apply(weights_init)

        self.opt_name = optimizer
        self.configure_optimizers()

        self.test_preds = {'preds': [], 'labels': [], 'stds': []}
        

    def test_epoch_end(self, outputs):
        for output in outputs:
            self.test_preds['preds'].extend(list(
                output['pred'].flatten().cpu().detach().numpy()))
            self.test_preds['labels'].extend(list(
                output['label'].cpu().detach().numpy()))
            self.test_preds['stds'].extend(list(
                output['std'].flatten().cpu().detach().numpy()))
    
    def configure_optimizers(self):
        """Because we use Pyro's SVI optimizer that works differently"""
        if self.opt_name == "radam":
            self.optimizer = pyro.optim.RAdam({"lr": self.lr})
        elif self.opt_name == "sgd":
            self.optimizer = pyro.optim.SGD({"lr": self.lr})
        elif self.opt_name == "adagrad": 
            self.optimizer = pyro.optim.Adagrad({"lr": self.lr})
        elif self.opt_name == "adam":
            self.optimizer = pyro.optim.ClippedAdam(
                {"lr": self.lr, "betas": (0.95, 0.999), 'clip_norm': 15})
        elif self.opt_name == "nadam":
            self.optimizer = pyro.optim.NAdam({"lr": self.lr})
        else:
            raise ValueError("Unknown optimizer")
        return None


    def on_save_checkpoint(self, checkpoint):
        """Saving Pyro's param_store for the bnn's parameters"""
        checkpoint["param_store"] = pyro.get_param_store().get_state()
        

    def on_load_checkpoint(self, checkpoint):
        pyro.get_param_store().set_state(checkpoint["param_store"])
        checkpoint['state_dict'] = remove_dict_entry_startswith(
            checkpoint['state_dict'], 'bnn')


    @staticmethod
    def add_model_specific_args(parent_parser):
        """To initialize from checkpoint, without giving init args """
        parser = parent_parser.add_argument_group("NCMAPSS_VIBnn")
        parser.add_argument("--net", type=str, default="inception")
        return parent_parser



class VIBnnWrapper(BnnWrapper):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        bias = True,
        archi="inception",
        activation="relu",
        optimizer='nadam',
        lr=1e-3,
        device=torch.device('cuda:0'),
        prior_loc=0.,
        prior_scale=10,
        likelihood_scale=3,
        q_scale=0.01,
        fit_context="flipout",
        guide_base="normal",
        num_particles=1,
        pretrain_file=None,
        last_layer=False,
        **kwargs
    ):
        super().__init__(
            win_length,
            n_features,
            dataset_size,
            bias=bias,
            archi=archi,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            device=device,
        )
        self.save_hyperparameters()
        closed_form_kl = True
        
        if pretrain_file is not None:
            sd = torch.load(pretrain_file, map_location=device)
            self.net.load_state_dict(sd)

        if fit_context == 'lrt':
            self.fit_ctxt = tyxe.poutine.local_reparameterization
        elif fit_context == 'flipout':
            self.fit_ctxt = tyxe.poutine.flipout
        else:
            self.fit_ctxt = contextlib.nullcontext

        if guide_base == 'normal':
            guide_base = tyxe.guides.AutoNormal
        elif guide_base == 'radial':
            guide_base = AutoRadial
            closed_form_kl = False
            print("Using Radial Guide")
            self.fit_ctxt = contextlib.nullcontext
        else: 
            raise RuntimeError("Guide unknown. Choose from 'normal', 'radial'.")
        
        self.num_particles = num_particles
        self.loss_name = "elbo"
        self.loss = (
            TraceMeanField_ELBO(num_particles) 
            if closed_form_kl
            else Trace_ELBO(num_particles) 
        )
        #self.loss = CustomTrace_ELBO(num_particles)
        prior_kwargs = {}#{'hide_all': True}
        
        self.prior = tyxe.priors.IIDPrior(
            dist.Normal(
                torch.tensor(float(prior_loc), device=device), 
                torch.tensor(float(prior_scale), device=device),
            ), **prior_kwargs
        )
        
        self.likelihood = tyxe.likelihoods.HeteroskedasticGaussian(
            dataset_size, positive_scale=False #scale=likelihood_scale
        )
        if pretrain_file is not None:
            print("Initializing weight distributions from pretrained net")
            self.net.load(pretrain_file, map_location=device)
            
            if not last_layer:
                self.guide = partial(
                    guide_base,
                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net),
                    init_scale=q_scale
                )
            else:
                print("Last_layer training only")
                for module in self.net.modules():
                    if module is not self.net.last: # -> last layer !
                        for param_name, param in list(module.named_parameters(recurse=False)):
                            delattr(module, param_name)
                            module.register_buffer(param_name, param.detach().data)
                
                self.guide = partial(guide_base,
                    init_loc_fn=tyxe.guides.PretrainedInitializer.from_net(self.net), 
                    init_scale=q_scale)

        else:
            if last_layer > 0:
                raise RuntimeError("No pretrain file but last_layer True")
            
            self.guide = partial(guide_base, init_scale=q_scale)
        #self.guide = None
        
        self.bnn = VariationalBNN(
            self.net, 
            self.prior, 
            self.likelihood, 
            self.guide,
            )
        self.svi = SVI(
            pyro.poutine.scale(self.bnn.model, scale=1./(dataset_size * win_length * n_features)),
            pyro.poutine.scale(self.bnn.guide, scale=1./(dataset_size * win_length * n_features)),
            self.optimizer,
            self.loss
        )


    def test_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1].squeeze()
        with self.fit_ctxt():
            output = self.bnn.predict(x, num_predictions=10)
            if isinstance(output, tuple):
                m, sd = output
            else:
                m, sd = output[:, 0], output[:, 1]
        
        mse = F.mse_loss(y.squeeze(), m.squeeze())
        self.log("mse/test", mse)

        try:
            return {"loss": mse, "label": batch[1], "pred": m.squeeze(), "std": sd}
        except NameError:
            return {"loss": mse, "label": batch[1], "pred": m.squeeze()}


    def validation_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1]
        
        # To compute only the KL part of the loss (no obs = no likelihood)
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(self.bnn_no_obs, self.bnn.guide,
            self.optimizer, self.loss)

        with self.fit_ctxt():
            elbo = self.svi.evaluate_loss(x, y.unsqueeze(-1))
            # Aggregate = False if num_prediction = 1, else nans in sd
            output = self.bnn.predict(x, num_predictions=1, aggregate=False)
            
            if isinstance(output, tuple):
                m, sd = output
            else:
                output = output.squeeze()
                m, sd = output[:, 0], output[:, 1]
            kl = self.svi_no_obs.evaluate_loss(x)
        
        mse = F.mse_loss(y.squeeze(), m.squeeze())

        self.log('elbo/val', elbo)
        self.log("mse/val", mse)
        self.log('kl/val', kl)
        self.log('likelihood/val', elbo - kl)
        #return {'loss' : mse}


    def training_step(self, batch, batch_idx):
        (x, y) = batch[0], batch[1]
        self.bnn_no_obs = pyro.poutine.block(self.bnn, hide=["obs"])
        self.svi_no_obs = SVI(self.bnn_no_obs, self.bnn.guide,
            self.optimizer, self.loss)

        # As we do not use PyTorch Optimizers, it is needed in order to Pytorch
        # Lightning to know that we are training a model, and initiate routines
        # like checkpointing etc.
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_ready()
        
        with self.fit_ctxt():
            elbo = self.svi.step(x, y.unsqueeze(-1))
            # Aggregate = False if num_prediction = 1, else nans in sd
            output = self.bnn.predict(x, num_predictions=1, aggregate=False)
            if isinstance(output, tuple):   # Homoscedastic
                m, sd = output
            else:                           # Heteroscedastic
                output = output.squeeze()
                m, sd = output[:, 0], output[:, 1]
            kl = self.svi_no_obs.evaluate_loss(x)
        self.trainer.fit_loop.epoch_loop.batch_loop.manual_loop.\
                                    optim_step_progress.increment_completed()
        
        
        mse = F.mse_loss(y.squeeze(), m.squeeze()).item()
        self.log("mse/train", mse)
        self.log("elbo/train", elbo)
        self.log('kl/train', kl)
        self.log('likelihood/train', elbo - kl)
        #return {'loss' : mse}





class MCMCBnnWrapper(BnnWrapper):
    def __init__(
        self,
        win_length,
        n_features,
        dataset_size,
        bias=True,
        archi='inception',
        activation='relu',
        optimizer='nadam',
        lr=1e-3,
        device=torch.device('cuda:0'),
        **kwargs
    ):
        super().__init__(
            win_length,
            n_features,
            dataset_size,
            bias=bias,
            archi=archi,
            activation=activation,
            optimizer=optimizer,
            lr=lr,
            device=device,
        )

        ...