import pytorch_lightning as pl

from bayesrul.ncmapss.dataset import NCMAPSSDataModule
from bayesrul.ncmapss.bayesian_models import NCMAPSSModelBnn
from optuna.integration import PyTorchLightningPruningCallback

import pyro
import argparse
import optuna

debug = False
EPOCHS = 2000 if not debug else 2

def objective(trial: optuna.trial.Trial) -> float:
    pyro.clear_param_store()

    print(f"TRIAL ID : {trial._trial_id}")

    prior_loc = trial.suggest_float("prior_loc", -0.2, 0.2)
    prior_scale = trial.suggest_float("prior_scale", 1e-5, 5, log=True)
    likelihood_scale = trial.suggest_float("likelihood_scale", 1e-5, 5, log=True)
    vardist_scale = trial.suggest_float("vardist_scale", 1e-5, 5, log=True)
    fit_context = trial.suggest_categorical("fit_context", ['lrt', 'flipout']) 
    lr = trial.suggest_float("lr", 1e-6, 1, log=True)
    args.archi = trial.suggest_categorical("args.archi", ['linear', 'conv'])
    #args.pretrain = trial.suggest_categorical("args.pretrain", [0, 10, 100])

    hyperparams = {
        'prior_loc' : prior_loc,
        'prior_scale' : prior_scale,
        'likelihood_scale' : likelihood_scale,
        'vardist_scale' : vardist_scale,
        'mode' : 'vi',
        'fit_context' : fit_context,
        'lr' : lr,
        'pretrain_file' : None,
        'trial_id' : trial._trial_id,
    }

    data = NCMAPSSDataModule(args.data_path, batch_size=10000)
    dnn = NCMAPSSModelBnn(data.win_length, data.n_features, data.train_size,
        archi = args.archi, **hyperparams)
    
    monitor = f"mse/val"
    trainer = pl.Trainer(
        gpus=[0],
        logger=True,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor=monitor)],
    )
    trainer.logger.log_hyperparams(hyperparams)
    trainer.fit(dnn, datamodule=data)

    return trainer.callback_metrics[monitor].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayesrul benchmarking')
    parser.add_argument('--data-path',
                    type=str,
                    default='data/ncmapss',
                    metavar='DATA',
                    help='Directory where to find the data')
    parser.add_argument('--out-path',
                    type=str,
                    default='results/ncmapss/',
                    metavar='OUT',
                    help='Directory where to store models and logs')
    parser.add_argument('--model-name',
                    type=str,
                    default='dnn',
                    metavar='NAME',
                    help='Name of this specific run. (default: dnn)')
    parser.add_argument('--archi',
                    type=str,
                    default='linear',
                    metavar='ARCHI',
                    help='Which model to run. (default: linear')
    parser.add_argument('--pretrain',
                        type=int,
                        metavar='PRETRAIN',
                        default=0,
                        help='Pretrain the BNN weights for x epoch. (default: 0)')

    args = parser.parse_args()
    
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.NopPruner()
    )
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))