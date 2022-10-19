from pathlib import Path
from typing import Optional

import hydra
import pyrootutils
from omegaconf import DictConfig

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import train
from optuna import Study
from optuna.trial import Trial

from .utils import get_pylogger

log = get_pylogger(__name__)


def objective_hnn(trial: Trial, cfg: DictConfig):
    cfg.datamodule.batch_size = trial.suggest_categorical(
        "batch_size", [100, 250, 500, 1000]
    )
    cfg.model.optimizer.lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    cfg.paths.output_dir = f"{cfg.paths.output_dir}/{trial.number:03d}"
    metric_dict, _ = train(cfg)
    return metric_dict[cfg.callbacks.model_checkpoint.monitor]


def objective_mcd(trial: Trial, cfg: DictConfig):
    cfg.model.mc_samples_train = trial.suggest_categorical(
        "mc_samples_train", [1, 2, 4, 6]
    )
    cfg.model.p_dropout = trial.suggest_float("p_dropout", 0.20, 0.85)
    return objective_hnn(trial, cfg)


def objective_bnn(trial: Trial, cfg: DictConfig):
    cfg.model.mc_samples_train = trial.suggest_categorical(
        "mc_samples_train", [1, 2, 4, 6]
    )
    cfg.model.prior_scale = trial.suggest_float(
        "prior_scale", 1e-2, 0.9, log=True
    )
    cfg.model.q_scale = trial.suggest_float("q_scale", 1e-4, 1e-1, log=True)
    return objective_hnn(trial, cfg)


@hydra.main(version_base=None, config_path="../conf", config_name="hpsearch")
def main(cfg: DictConfig) -> Optional[float]:

    log.info(f"Instantiating study <{cfg.hpsearch.study._target_}>")
    path = Path(f"{cfg.paths.root_dir}/results")
    study: Study = hydra.utils.instantiate(
        cfg.hpsearch.study,
        storage=f"sqlite:///{path.as_posix()}/{cfg.hpsearch.study.study_name}.db",
    )
    log.info(f"Instantiating objective <{cfg.hpsearch.objective._target_}>")
    objective = hydra.utils.instantiate(cfg.hpsearch.objective, _partial_=True)

    log.info(f"Starting hyperparameter search ...")
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=1,
        timeout=None,
        catch=(RuntimeError,),
    )

    log.info("Number of finished trials: {}".format(len(study.trials)))


if __name__ == "__main__":
    main()
