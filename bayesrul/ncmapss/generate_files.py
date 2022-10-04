import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .preprocessing import generate_lmdb, preprocess_lmdb


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.ds.data_path = to_absolute_path(cfg.ds.data_path)
    cfg.ds.out_path = to_absolute_path(cfg.ds.out_path)
    generate_lmdb(cfg.ds)
    preprocess_lmdb(cfg.ds)


if __name__ == "__main__":
    main()
