import hydra
from pathlib import Path
import pyrootutils
from omegaconf import DictConfig

from ..results.predictions import load_predictions

from .utils import get_pylogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="genres")
def main(cfg: DictConfig) -> None:
    predictions = load_predictions(
        cfg.methods, cfg.de_base_learners, cfg.runs_dir, cfg.cache_dir
    )


if __name__ == "__main__":
    main()
