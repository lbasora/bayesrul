import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .preprocessing import generate_lmdb, preprocess_lmdb


@hydra.main(version_base=None, config_path="../conf", config_name="gen_files")
def main(cfg: DictConfig) -> None:
    cfg.dataset.data_path = to_absolute_path(cfg.dataset.data_path)
    cfg.dataset.out_path = to_absolute_path(cfg.dataset.out_path)
    generate_lmdb(cfg.dataset)
    preprocess_lmdb(cfg.dataset)


if __name__ == "__main__":
    main()
