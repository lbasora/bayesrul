import hydra
from pathlib import Path
import pyrootutils
from omegaconf import DictConfig

from ..results.predictions import load_predictions
from ..results.plots import plot_metrics
from ..results.metrics import model_metrics, method_metrics, ds_metrics
from ..results.latex import df_to_latex, to_mean_std

from .utils import get_pylogger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

log = get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="figs")
def main(cfg: DictConfig) -> None:
    log.info("Computing metrics from model predictions ...")
    df_preds = load_predictions(
        cfg.methods,
        cfg.de_base_learners,
        f"{cfg.paths.data_dir}/{cfg.dataset}",
        cfg.runs_dir,
        cfg.cache_dir,
    )
    df_model = model_metrics(df_preds, cfg.metrics, top=5)
    df_method = method_metrics(df_model, cfg.metrics)
    df_ds = ds_metrics(df_preds, cfg.metrics)

    log.info("Generating plots ...")
    Path(cfg.figs_dir).mkdir(exist_ok=True)
    plot_metrics(
        df_model,
        "method",
        [0.75, 0.8],
        f"{cfg.figs_dir}/method_metrics.pdf",
        legend=False,
    )
    plot_metrics(
        df_ds,
        "dataset",
        [0.90, 0.80],
        f"{cfg.figs_dir}/ds_metrics.pdf",
        legend=True,
    )

    if cfg.csv_tables:
        log.info("Generating CSV files ...")
        df_model.to_csv(f"{cfg.figs_dir}/df_model.csv")
        df_method.to_csv(f"{cfg.figs_dir}/df_method.csv")
        df_ds.to_csv(f"{cfg.figs_dir}/df_ds.csv")

    if cfg.latex:
        log.info("Generating latex files ...")
        df_to_latex(df_model, cfg.figs_dir, "df_model", pdf=cfg.latex_pdf)
        df_to_latex(
            to_mean_std(df_method, cfg.metrics).loc[cfg.methods].reset_index(),
            cfg.figs_dir,
            "df_method",
            highlight_min=False,
            pdf=cfg.latex_pdf,
        )


if __name__ == "__main__":
    cfg = main()
