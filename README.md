# Bayesrul

This is a library for the benchmark of some uncertainty quantification methods (UQ) for deep learning (DL) in the context of the Remaining Useful Life (RUL) prognostics. We experiment with deterministic heteroscedastic neural networks (HNN), deep ensembles (DE), Monte Carlo dropout (MCD) and several Bayesian Neural Networks (BNN) techniques for variance reduction such as local reparametrization trick (LRT), Flipout (FO) and Radial Bayesian networks (RAD).

The dataset we use for the benchmark (N-CMAPSS) needs to be downloaded from the NASA prognostics website. Three deep neural networks are implemented with pytorch and pytorch-lightning and turned into BNN with pyro and TyXe. Hyperparameter search is implemented with Optuna. The library computes negative log-likelihood (NLL), root mean squared error (RMSE), root mean sqaured calibration error (RMSCE) and sharpness to evaluate aspects such model accuracy and quality of predictive uncertainty estimation. Plots and csv files are generated for analysis.

The library uses poetry for dependency management and hydra for configuration management. 
## Setup 

Clone the repository
```
git clone git@github.com:lbasora/bayesrul.git
cd bayesrul
```

Use poetry to install dependencies
```
poetry install
```
## Using the library
The library uses hydra library and the conf files are in `bayesruls/conf/`
#### Generate dataset lmdb files for N-CMAPSS
Make sure to have the downloaded N-CMAPSS files from NASA at `data/ncmapss/`
```
poetry run python -m bayesrul.tasks.build_ds
```
You can overload the options in the conf file  `bayesruls/conf/build_ds.yaml`

#### Hyperparameter search
Example with a HNN:
```
poetry run python -m bayesrul.tasks.hpsearch hpsearch=ncmapss_hnn task_name=hps_hnn
```

#### Model training and test

```
poetry run python -m bayesrul.tasks.train experiment=ncmapss_hnn task_name=train_hnn
```

For instance, if you want to execute 5 runs of the HNN model with different seeds by exploiting hydra multirun facility:

```
poetry run python -m bayesrul.tasks.train experiment=ncmapss_hnn seed=1,2,3,4,5 task_name=train_hnn --multirun
```

#### Model evaluation
```
poetry run python -m bayesrul.tasks.metrics
```

## References
TODO

