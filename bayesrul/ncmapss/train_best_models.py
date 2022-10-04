import argparse
import json
from pathlib import Path

import torch

from bayesrul.inference.deep_ensemble import DeepEnsemble
from bayesrul.inference.dnn import HeteroscedasticDNN
from bayesrul.inference.mc_dropout import MCDropout
from bayesrul.inference.vi_bnn import VI_BNN
from bayesrul.ncmapss.dataset import NCMAPSSDataModule

"""
For a given model ("FLIPOUT" for example), retrieves the best parameters in the file
results/ncmapss/best_models/FLIPOUT/000.json and trains a model according
to the hyperparameters stored in this file

Deep ensembles train more epochs, because training examples are asplit among ensembles
"""


def isbayesian(s):
    # s = "_".join(s.split("_")[:-1])
    if s.upper() in ["MFVI", "RADIAL", "LOWRANK", "LRT", "FLIPOUT"]:
        return True
    elif s.upper() in ["MC_DROPOUT", "DEEP_ENSEMBLE", "HETERO_NN"]:
        return False
    else:
        raise ValueError(f"Unknow model {s}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesrul benchmarking")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ncmapss",
        metavar="DATA",
        help="Directory where to find the data",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="results/ncmapss/",
        metavar="OUT",
        help="Directory where to store models and logs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LRT",
        metavar="MODEL",
        required=True,
        help="Model and name (ex: LRT_001)",
    )
    parser.add_argument(
        "--GPU",
        type=int,
        default="results/ncmapss/",
        metavar="GPU",
        required=True,
        help="GPU index (ex: 1)",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        metavar="EARLY_STOP",
        required=True,
        help="Early stopping patience (default:0 early stooping disabled",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="EPOCHS",
        required=True,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1e4,
        metavar="BATCH_SIZE",
        required=True,
        help="Batch size (default: 10000)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        metavar="NUM_RUNS",
        required=True,
        help="Number of runs (default: 1)",
    )

    args = parser.parse_args()

    model = "_".join(args.model.split("_")[:-1])
    model_path = Path("results/ncmapss/best_models", model)
    ls = sorted(list(model_path.glob("*.json")))

    with open(ls[0], "r") as f:
        hyp = json.load(f)
    try:
        del hyp["value_0"]
        del hyp["value_1"]
    except KeyError:
        pass

    data = NCMAPSSDataModule(args.data_path, batch_size=args.batch_size)
    monitor = "elbo/val" if isbayesian(model) else "gaussian_nll/val"

    run_int = int(args.model.split("_")[-1])
    for i in range(run_int, run_int + args.num_runs):
        args.model_name = model + "_" + f"{i:03}"

        if args.num_runs > 1:
            print(f"--------------- {args.model_name} -------------------")

        if isbayesian(model):
            hyp["pretrain"] = 5
            module = VI_BNN(args, data, hyp, GPU=args.GPU)
        else:
            if model == "MC_DROPOUT":
                p_dropout = hyp["p_dropout"]
                module = MCDropout(
                    args, data, hyp["p_dropout"], hyp, GPU=args.GPU
                )
            elif model == "DEEP_ENSEMBLE":
                module = DeepEnsemble(
                    args, data, hyp["n_models"], hyp, GPU=args.GPU
                )
            elif model == "HETERO_NN":
                module = HeteroscedasticDNN(args, data, hyp, GPU=args.GPU)
            else:
                raise ValueError(
                    f"Wrong model {model}. Available : MFVI, "
                    "RADIAL, LOWRANK, MC-DROPOUT, DEEP-ENSEMBLE, HETERO-NN"
                )

        module.fit(args.epochs, monitor=monitor, early_stop=args.early_stop)
        module.test()
        try:
            module.epistemic_aleatoric_uncertainty(device=torch.device("cpu"))
        except Exception:
            pass
