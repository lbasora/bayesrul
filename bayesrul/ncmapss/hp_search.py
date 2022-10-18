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

from optuna import Study
from optuna.trial import Trial

from bayesrul.ncmapss.train import train
from bayesrul.utils import get_pylogger

log = get_pylogger(__name__)


class HNN:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def __call__(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        self.cfg.model.optimizer.lr = lr
        metric_dict, _ = train(self.cfg)
        # return metric_dict[self.cfg.optimized_metric]
        return metric_dict[self.cfg.callbacks.model_checkpoint.monitor]


class MCD(HNN):
    def __call__(self, trial):
        p_dropout = trial.suggest_float("p_dropout", 0.20, 0.85)
        self.cfg.model.p_dropout = p_dropout
        return super().__call__(trial)


@hydra.main(version_base=None, config_path="../conf", config_name="hp_search")
def main(cfg: DictConfig) -> Optional[float]:
    # metric_dict, _ = train(cfg)
    # return metric_dict[cfg.optimized_metric]

    path = Path(f"{cfg.paths.root_dir}/results")
    # path.mkdir(exist_ok=True, parents=True)

    log.info(f"Instantiating study <{cfg.hp_search.study._target_}>")
    study: Study = hydra.utils.instantiate(
        cfg.hp_search.study,
        storage=f"sqlite:///{path.as_posix()}/{cfg.hp_search.study.study_name}.db",
    )
    objective = hydra.utils.instantiate(cfg.hp_search.objective, cfg)

    study.optimize(
        objective,
        n_trials=1,
        timeout=None,
        catch=(RuntimeError,),
    )

    log.info("Number of finished trials: {}".format(len(study.trials)))


if __name__ == "__main__":
    main()
