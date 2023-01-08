import hydra
from pathlib import Path
import pyrootutils
from omegaconf import DictConfig

from ..results.predictions import load_predictions
from ..results.metrics import Metrics

from .utils import get_pylogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="metrics")
def main(cfg: DictConfig) -> None:
    log.info(f"Computing metrics for {cfg.subset} set...")
    df_preds = load_predictions(
        f"{cfg.cache_dir}/predictions.parquet",
        cfg.methods,
        cfg.deepens,
        f"{cfg.paths.data_dir}/{cfg.dataset}",
        f"{cfg.preds_dir}",
        f"{cfg.cache_dir}",
        cfg.subset,
    )
    metrics = Metrics(df_preds, cfg.metrics, cfg.methods, top=cfg.top)
    metrics_dict = dict()
    # metrics_dict["dataset"] = metrics.by_dataset()
    if cfg.subset == "test":
        # metrics_dict["method"] = metrics.by_method()
        # metrics_dict["method_agg"] = metrics.by_method(agg=True)
        # metrics_dict["dataset_agg"] = metrics.by_dataset(agg=True)
        # metrics_dict["unit"] = metrics.by_unit()
        # metrics_dict["unit_agg"] = metrics.by_unit(agg=True)
        # metrics_dict["unit_method"] = metrics.by_unit_method()
        # metrics_dict["fc"] = metrics.by_fc()
        # metrics_dict["fc_agg"] = metrics.by_fc(agg=True)
        # metrics_dict["fc_method"] = metrics.by_fc_method()
        # metrics_dict["calibration"] = metrics.calibration()
        # metrics_dict["calibration_fc1"] = metrics.calibration(fc=1)
        # metrics_dict["calibration_fc2"] = metrics.calibration(fc=2)
        # metrics_dict["calibration_fc3"] = metrics.calibration(fc=3)
        # metrics_dict["acc_vs_conf"] = metrics.acc_vs_conf()
        metrics_dict["rlt_method"] = metrics.by_rlt_method()
        metrics_dict[
            "rlt_unit_method_std_err"
        ] = metrics.by_rlt_unit_method_std_err()
        metrics_dict["rlt_unit_method_rul"] = metrics.by_rlt_unit_method_rul()
        metrics_dict[
            "rlt_unit_method_eps_al"
        ] = metrics.by_rlt_unit_method_eps_al()
    path = f"{cfg.metrics_dir}/csv"
    Path(path).mkdir(exist_ok=True, parents=True)
    for key, value in metrics_dict.items():
        value.to_csv(f"{path}/{key}.csv")


if __name__ == "__main__":
    cfg = main()
