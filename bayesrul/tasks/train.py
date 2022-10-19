from pathlib import Path
from typing import List, Optional

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.loggers import LightningLoggerBase

from ..models.frequentist import NN
from ..utils.miscellaneous import ResultSaver
from .utils import (
    get_pylogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)


@task_wrapper
def train(cfg: DictConfig):
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    cfg.model.net.win_length = datamodule.win_length
    cfg.model.net.n_features = datamodule.n_features
    if OmegaConf.is_missing(cfg.model, "dataset_size"):
        cfg.model.dataset_size = datamodule.train_size
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, _convert_="partial"
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        if cfg.model.get("pretrain_epochs") and not cfg.get("ckpt_path"):
            model.net = load_pretrained_net(cfg, model)
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning(
                "Best ckpt not found! Using current weights for testing..."
            )
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")
        log.info(f"Saving predicions: {cfg.paths.output_dir}/predictions/")
        results = ResultSaver(f"{cfg.paths.output_dir}")
        results.save(model.test_preds)

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def load_pretrained_net(cfg, model):
    if cfg.model.pretrain_epochs == 0:
        return hydra.utils.instantiate(cfg.model.net)
    ckpt_path_dir = Path(f"{cfg.paths.results_dir}/{cfg.tags[0]}/pretrained")
    if ckpt_path_dir is not None:
        ckpt_path = None
        for d in ckpt_path_dir.glob("*"):
            try:
                i = int(d.name)
            except:
                continue
            if i + 1 == cfg.model.pretrain_epochs:
                ckpt_path = Path(d, "checkpoints", "pretrained.ckpt")
                break
        if ckpt_path is not None:
            log.info(f"Restoring pretrained net from: {ckpt_path}")
            return NN.load_from_checkpoint(
                ckpt_path,
                net=model.net,
            ).net
    log.info(
        f"No pretrained net found in {ckpt_path_dir} for {cfg.model.pretrain_epochs}"
    )
    return hydra.utils.instantiate(cfg.model.net)


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    train(cfg)


if __name__ == "__main__":
    main()
