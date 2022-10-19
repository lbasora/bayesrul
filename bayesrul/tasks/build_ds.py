import hydra
import pyrootutils
from omegaconf import DictConfig

from .utils import get_pylogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="build_ds")
def main(cfg: DictConfig) -> None:
    log.info(f"Instantiating dataset builder <{cfg.dataset.builder._target_}>")
    hydra.utils.instantiate(cfg.dataset.builder, cfg)()


if __name__ == "__main__":
    main()
