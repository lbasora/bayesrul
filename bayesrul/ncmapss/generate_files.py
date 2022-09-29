import hydra
from omegaconf import DictConfig, OmegaConf
from types import SimpleNamespace
from .preprocessing import (
    generate_parquet,
    generate_lmdb,
    generate_unittest_subsample,
)


@hydra.main(
    version_base=None, config_path="../conf", config_name="generate_files"
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.out_path)
    generate_parquet(cfg)
    generate_lmdb(cfg, datasets=["train", "test"])

    # args = SimpleNamespace(
    #     out_path="data/ncmapss/",
    #     test_path="tests/",
    #     validation=0,
    #     files=[
    #         "N-CMAPSS_DS01-005",
    #         "N-CMAPSS_DS02-006",
    #         "N-CMAPSS_DS03-012",
    #         "N-CMAPSS_DS04",
    #         "N-CMAPSS_DS05",
    #     ],
    #     subdata=["X_s", "A"],
    #     moving_avg=False,  # Smooth the values of the sensors
    #     win_length=30,  # Window size
    #     win_step=10,  # Window step
    #     skip_obs=10,  # How much to downsample the huge dataset
    #     bits=32,  # Size of numbers in memory
    #     lmdb_min_max=False,
    # )

    # generate_parquet(args)
    # generate_lmdb(args, datasets=["train", "test"])

    # args.files = ["N-CMAPSS_DS02-006"]
    # generate_unittest_subsample(args)  # To create unit test parquets


if __name__ == "__main__":
    main()
