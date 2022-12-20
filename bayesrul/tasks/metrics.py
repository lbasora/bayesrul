import hydra
from pathlib import Path
import pyrootutils
from omegaconf import DictConfig

from ..results.predictions import load_predictions, smooth_cols
from ..results.plots import (
    plot_metrics,
    plot_unit_method_std_err,
    plot_rul,
    plot_eps_al_uncertainty,
    plot_calibration,
)
from ..results.metrics import Metrics
from ..results.latex import df_to_latex, to_mean_std

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
    log.info("Computing metrics ...")
    df_preds = load_predictions(
        cfg.methods,
        cfg.deepens,
        f"{cfg.paths.data_dir}/{cfg.dataset}",
        cfg.preds_dir,
        cfg.cache_dir,
        cfg.subset,
    )
    metrics = Metrics(df_preds, cfg.metrics, cfg.methods, top=cfg.top)
    # df_best_model = metrics.by_best_model(metric="nll")
    # df_preds_best = smooth_cols(df_preds, df_best_model, cfg.cache_dir)
    metrics_dict = dict()
    # metrics_dict["df_method"] = metrics.by_method()
    # metrics_dict["df_method_agg"] = metrics.by_method(agg=True)
    metrics_dict["df_dataset"] = metrics.by_dataset()
    # metrics_dict["df_dataset_agg"] = metrics.by_dataset(agg=True)
    # metrics_dict["df_unit"] = metrics.by_unit()
    # metrics_dict["df_unit_agg"] = metrics.by_unit(agg=True)
    metrics_dict["df_fc"] = metrics.by_fc()
    metrics_dict["df_fc_agg"] = metrics.by_fc(agg=True)

    path = f"{cfg.metrics_dir}/fig"
    log.info(f"Generating plots in {path} ...")
    Path(path).mkdir(exist_ok=True, parents=True)
    # plot_metrics(
    #     metrics_dict["df_method"],
    #     "method",
    #     [0.75, 0.8],
    #     f"{path}/method_metrics.pdf",
    #     legend=False,
    # )
    plot_metrics(
        metrics_dict["df_dataset"],
        "dataset",
        [0.90, 0.80],
        f"{path}/ds_metrics.pdf",
        legend=False,
    )
    plot_metrics(
        metrics_dict["df_fc"],
        "fc",
        [0.90, 0.85],
        f"{path}/fc_metrics.pdf",
        legend=True,
    )
    # plot_metrics(
    #     metrics_dict["df_unit"],
    #     "unit",
    #     [0.9, 0.8],
    #     f"{path}/unit_metrics.pdf",
    #     aspect=2.5,
    #     label_rotation=0,
    # )
    # plot_unit_method_std_err(
    #     df_preds_best,
    #     cfg.methods,
    #     f"{path}/unit_method_std_err_nll.pdf",
    # )
    # plot_rul(
    #     df_preds_best,
    #     cfg.methods,
    #     cfg.rul_units,
    #     f"{path}/rul.pdf",
    # )
    # plot_eps_al_uncertainty(
    #     df_preds_best,
    #     cfg.ep_al_methods,
    #     cfg.rul_units,
    #     f"{path}/eps_al_uq.pdf",
    # )
    # plot_calibration(
    #     df_preds_best,
    #     cfg.methods,
    #     f"{path}/model_calibration.pdf",
    # )

    path = f"{cfg.metrics_dir}/csv"
    log.info(f"Generating CSV files in {path} ...")
    Path(path).mkdir(exist_ok=True, parents=True)
    for key, value in metrics_dict.items():
        value.to_csv(f"{path}/{key}.csv")

    if cfg.latex:
        path = f"{cfg.metrics_dir}/latex"
        log.info("Generating latex files in {path} ...")
        Path(path).mkdir(exist_ok=True, parents=True)
        for key, value in metrics_dict.items():
            if key.endswith("agg"):
                df_to_latex(
                    to_mean_std(value, cfg.metrics).reset_index(),
                    path,
                    key,
                    highlight_min=False,
                )


if __name__ == "__main__":
    cfg = main()
