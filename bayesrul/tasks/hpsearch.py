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

from .train import train
from optuna import Study
from optuna.trial import Trial

from .utils import get_pylogger, instantiate_callbacks

log = get_pylogger(__name__)
pruner_in_callbacks = False


def objective(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.datamodule.batch_size = trial.suggest_categorical(
        "batch_size", [100, 250, 500, 1000]
    )
    log.info(f"{cfg.datamodule.batch_size} batch_size")
    log.info(
        f"_________________ Starting trial {trial.number:03d} __________________"
    )
    cfg.paths.output_dir = f"{output_dir}/{trial.number:03d}"

    global pruner_in_callbacks
    callbacks = None
    if not pruner_in_callbacks:
        log.info("Instantiating callbacks (with optuna pruner)...")
        callbacks = instantiate_callbacks(
            cfg.get("callbacks"), exclude=["optuna_pruner"]
        )
        callbacks.append(
            hydra.utils.instantiate(
                cfg.callbacks.optuna_pruner,
                trial=trial,
                monitor=cfg.hpsearch.monitor,
            )
        )
        pruner_in_callbacks = True
    metric_dict, _ = train(cfg, callbacks)
    return metric_dict[cfg.hpsearch.monitor]


def objective_hnn(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.model.optimizer.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    log.info(f"{cfg.model.optimizer.lr} lr")
    return objective(trial, cfg, output_dir)


def objective_mcd(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.model.mc_samples = trial.suggest_categorical(
        "mc_samples", [20, 50, 100]
    )
    log.info(f"{cfg.model.mc_samples} mc_samples")
    cfg.model.p_dropout = trial.suggest_float("p_dropout", 0.20, 0.85)
    log.info(f"{cfg.model.p_dropout} p_dropout")
    return objective_hnn(trial, cfg, output_dir)


def objective_bnn(trial: Trial, cfg: DictConfig, output_dir: str):
    cfg.model.pretrain_epochs = trial.suggest_categorical(
        "pretrain_epochs", [0, 5]
    )
    log.info(f"{cfg.model.pretrain_epochs} pretrain_epochs")
    cfg.model.optimizer._args_[0]["lr"] = trial.suggest_float(
        "lr", 1e-5, 1e-3, log=True
    )
    log.info(f"{cfg.model.optimizer._args_[0]['lr']} lr")
    cfg.model.mc_samples_train = trial.suggest_categorical(
        "mc_samples_train", [1, 2]
    )
    log.info(f"{cfg.model.mc_samples_train} mc_samples_train")
    cfg.model.prior_scale = trial.suggest_float(
        "prior_scale", 1e-2, 0.5, log=True
    )
    log.info(f"{cfg.model.prior_scale} prior_scale")
    cfg.model.q_scale = trial.suggest_float("q_scale", 1e-4, 1e-2, log=True)
    log.info(f"{cfg.model.q_scale} q_scale")
    return objective(trial, cfg, output_dir)


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

    output_dir = cfg.paths.output_dir
    log.info(f"Starting hyperparameter search ...")
    study.optimize(
        lambda trial: objective(trial, cfg, output_dir),
        n_trials=cfg.hpsearch.n_trials,
        timeout=None,
        catch=(RuntimeError,),
    )

    log.info("Number of finished trials: {}".format(len(study.trials)))


if __name__ == "__main__":
    main()
