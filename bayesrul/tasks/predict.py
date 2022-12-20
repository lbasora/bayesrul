from typing import Optional

import hydra
import pyrootutils
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

import pandas as pd
from pathlib import Path

from ..utils.miscellaneous import ResultSaver
from .utils import get_pylogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)


def predict(cfg: DictConfig):
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    ckpt_paths = []
    if cfg.ckpt_path == "all":
        path = Path(f"{cfg.runs_dir}")
        for ckpt_path in path.glob("**/epoch*.ckpt"):
            ckpt_paths.append(ckpt_path)
    else:
        ckpt_paths.append(Path(cfg.ckpt_path))

    for ckpt_path in ckpt_paths:
        method, run = ckpt_path.as_posix().split("/")[-4:-2]
        run = f"{int(run):03d}"

        model = cfg[method]
        log.info(f"Instantiating model <{model._target_}>")
        model.net.win_length = datamodule.win_length
        model.net.n_features = datamodule.n_features
        if OmegaConf.is_missing(model, "dataset_size"):
            model.dataset_size = datamodule.train_size
        model: LightningModule = hydra.utils.instantiate(
            model, _convert_="partial"
        )

        for subset in cfg.subsets:
            filename = f"{method}_{run}_{subset}.parquet"
            datamodule.set_predict_dataset(subset)
            predictions = trainer.predict(
                model=model, datamodule=datamodule, ckpt_path=ckpt_path
            )
            predictions = pd.DataFrame.from_records(predictions)
            predictions = predictions.explode(
                predictions.columns.tolist()
            ).reset_index(drop=True)
            log.info(f"Saving predicions: {cfg.paths.output_dir}")
            results = ResultSaver(f"{cfg.paths.output_dir}", f"{filename}")
            results.save(predictions)


@hydra.main(version_base=None, config_path="../conf", config_name="predict")
def main(cfg: DictConfig) -> Optional[float]:
    predict(cfg)


if __name__ == "__main__":
    main()
