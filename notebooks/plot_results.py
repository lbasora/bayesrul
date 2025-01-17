# To be executed with VS Code jupyter extension

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pyrootutils
import seaborn as sns
from hydra import compose, initialize

import pandas as pd
from bayesrul.results.latex import latex_formatted, to_mean_std
from bayesrul.results.metrics import compute_cdf
from bayesrul.results.plots import *
from bayesrul.results.predictions import load_predictions

sns.set_style("white")
sns.set_context("paper")
sns.despine()
tex_fonts = {
    "text.usetex": True,
    "font.family": "Ubuntu",
    "axes.labelsize": 14,
    "font.size": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}
plt.rcParams.update(tex_fonts)

metrics = ["mae", "rmse", "nll", "ece", "sharp", "s"]

# %%
root = pyrootutils.setup_root(
    search_from="..",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
initialize(version_base=None, config_path="../bayesrul/conf")
cfg = compose(
    config_name="metrics",
    return_hydra_config=True,
)
path_fig = f"{cfg.metrics_dir}/fig"
Path(path_fig).mkdir(exist_ok=True, parents=True)

# %%
df = load_predictions(
    f"{cfg.cache_dir}/predictions.parquet",
    cfg.methods,
    cfg.deepens,
    f"{cfg.paths.data_dir}/{cfg.dataset}",
    f"{cfg.preds_dir}",
    f"{cfg.cache_dir}",
    cfg.subset,
)
df.head()

# %% [markdown]
# ## Method metrics
# %%
print(
    latex_formatted(
        to_mean_std(
            pd.read_csv(f"{cfg.metrics_dir}/csv/method_agg.csv"),
            metrics=metrics,
            precision=2,
        ),
        highlight_min=False,
    )
)

# %%
plot_metrics(
    pd.read_csv(f"{cfg.metrics_dir}/csv/method.csv"),
    "method",
    [0.75, 0.8],
    save_as=f"{path_fig}/method_metrics.pdf",
    legend=False,
)

# %% [markdown]
# ## DS/Unit metrics

# %%
print(
    latex_formatted(
        to_mean_std(
            pd.read_csv(f"{cfg.metrics_dir}/csv/dataset_agg.csv"),
            metrics=metrics,
            precision=2,
        ),
        highlight_min=False,
    )
)

# %%
print(
    latex_formatted(
        to_mean_std(
            pd.read_csv(f"{cfg.metrics_dir}/csv/unit_agg.csv"),
            metrics=metrics,
        ),
        highlight_min=False,
    )
)

# %%
plot_unit_method_metrics(
    pd.read_csv(f"{cfg.metrics_dir}/csv/unit_method.csv"),
    cfg.methods,
    xticklabel_size=13,
    metric_label_size=20,
    legend_label_size=18,
    save_as=f"{path_fig}/unit_method_metrics.pdf",
)

# %% [markdown]
# ## Flight Class metrics


# %%
plot_fc_method_metrics(
    pd.read_csv(f"{cfg.metrics_dir}/csv/fc_method.csv"),
    cfg.methods,
    bbox=[0.45, 1.04],
    xticklabel_size=30,
    save_as=f"{path_fig}/fc_method_metrics.pdf",
)

# %% [markdown]
# ## Metrics per DS for test and training sets (part of D4 analysis)
# %%
cfg = compose(config_name="metrics", overrides=["subset=test"])
plot_metrics(
    pd.read_csv(f"{cfg.metrics_dir}/csv/dataset.csv"),
    "dataset",
    [0.90, 0.80],
    save_as=f"{path_fig}/ds_metrics.pdf",
    legend=False,
)

# %%
cfg = compose(config_name="metrics", overrides=["subset=train"])
plot_metrics(
    pd.read_csv(f"{cfg.metrics_dir}/csv/dataset.csv"),
    "dataset",
    [0.90, 0.80],
    f"{path_fig}/ds_metrics_train.pdf",
    aspect=1,
    legend=True,
)

# %% [markdown]
# ## Calibration
# %%
# To generate tikz plot illustrating over/under confidence
cfg = compose(config_name="metrics", overrides=["subset=test"])
df = pd.read_csv(f"{cfg.metrics_dir}/csv/calibration.csv")
uc = df.query("model =='MCD_000'")[["exp_c", "obs_c"]]
oc = df.query("model == 'FO_000'")[["exp_c", "obs_c"]]
for cal, df in zip(["uc", "oc"], [uc, oc]):
    df.rename(columns={"exp_c": "x", "obs_c": "y"}).to_csv(
        f"{cal}_cal.csv", sep=" ", index=False, header=False
    )

# %%
cfg = compose(config_name="metrics", overrides=["subset=test"])
plot_calibration(
    pd.read_csv(f"{cfg.metrics_dir}/csv/calibration.csv"),
    cfg.methods,
    f"{path_fig}/calibration.pdf",
)

# %%
plot_calibration(
    pd.read_csv(f"{cfg.metrics_dir}/csv/calibration_fc1.csv"),
    cfg.methods,
    f"{path_fig}/calibration_fc1.pdf",
)

# %%
plot_calibration(
    pd.read_csv(f"{cfg.metrics_dir}/csv/calibration_fc3.csv"),
    cfg.methods,
    f"{path_fig}/calibration_fc3.pdf",
)

# %% [markdown]
# ## Accuracy vs confidence
# %%
plot_acc_vs_conf(
    pd.read_csv(f"{cfg.metrics_dir}/csv/acc_vs_conf.csv"),
    f"{path_fig}/calibration_fc3.pdf",
)

# %% [markdown]
# ## OOD and D4 analysis
# %%
import torch

df = df.assign(
    entropy=lambda x: torch.distributions.normal.Normal(
        torch.tensor(x.labels.values), torch.tensor(x.stds.values)
    )
    .entropy()
    .numpy()
)

# %% [markdown]
# ### D4 analysis
# %%
ood_units = ["D4U08", "D4U10", "D4U09", "D4U07"]
df_ood = df.query(f"unit.isin({ood_units})").assign(ood_id="OOD")
df_id = df.query(f"~unit.isin({ood_units})").assign(ood_id="ID")
df_ood_id = pd.concat([df_ood, df_id])

# %%
ood_boxplot(
    df_ood_id,
    save_as=f"{path_fig}/entropy_boxplot_d4.pdf",
    ylim=[0, 5.5],
    leg_labels=["D4", "Other test units"],
)

# %%
ood_cdfplot(
    compute_cdf(df_ood_id),
    cfg.methods,
    save_as=f"{path_fig}/entropy_cdf_d4.pdf",
    xlim=[-1, 5.5],
    leg_labels=["D4", "Other test units"],
)

# %% [markdown]
# ### OOD units

# %%
ood_units = ["D2U11", "D3U11", "D3U12"]  # , "D4U09"]
df_ood = df.query(f"unit.isin({ood_units})").assign(ood_id="OOD")
df_id = df.query(
    f"~unit.str.startswith('D4') and ~unit.isin({ood_units})"
).assign(ood_id="ID")
df_ood_id = pd.concat([df_ood, df_id])

# %%
ood_boxplot(
    df_ood_id, save_as=f"{path_fig}/entropy_boxplot_ood.pdf", ylim=[-0.5, 5.5]
)

# %%
ood_cdfplot(
    compute_cdf(df_ood_id),
    cfg.methods,
    save_as=f"{path_fig}/entropy_cdf_ood.pdf",
    xlim=[-1, 5],
)

# %%
plot_rlt_metrics(
    pd.read_csv(f"{cfg.metrics_dir}/csv/rlt_method.csv"),
    cfg.methods,
    save_as=f"{path_fig}/rlt_metrics.pdf",
)


# %%
df_rlt = pd.read_csv(f"{cfg.metrics_dir}/csv/rlt_unit_method_std_err.csv")
units = (
    df_rlt.groupby("unit")
    .mean(numeric_only=True)
    .sort_values("stds", ascending=False)
    .reset_index()
    .unit.unique()
    .tolist()
)
grey_units = units[5:-5]
units = units[:5] + units[-5:]
plot_rlt_unit_method_std_err(
    df_rlt,
    ["LRT", "FO", "RAD", "MCD", "DE", "HNN"],
    save_as=f"{path_fig}/unit_method_std_err.pdf",
    units=units,
    grey_units=grey_units,
    xticklabel_size=20,
    method_label_size=30,
    legend_label_size=30,
)

# %%
plot_rlt_unit_method_rul(
    pd.read_csv(f"{cfg.metrics_dir}/csv/rlt_unit_method_rul.csv"),
    methods=["LRT", "FO", "RAD", "MCD", "DE", "HNN"],
    units=["D4U08", "D5U08"],
    save_as=f"{path_fig}/rul.pdf",
    xticklabel_size=18,
    method_label_size=20,
    legend_label_size=18,
)

# %%
plot_eps_al_uncertainty(
    pd.read_csv(f"{cfg.metrics_dir}/csv/rlt_unit_method_eps_al.csv"),
    methods=["LRT", "FO", "RAD", "MCD"],
    units=["D4U08", "D5U08"],
    save_as=f"{path_fig}/eps_al_uq.pdf",
    xticklabel_size=20,
    method_label_size=20,
    legend_label_size=20,
)

# %%
