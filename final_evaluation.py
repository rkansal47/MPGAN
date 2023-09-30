import json
import os

import jetnet
from jetnet.datasets import JetNet
from jetnet import evaluation

import numpy as np


real_samples = {}
num_samples = 50000
num_w1_eval_samples = 10000
num_w1_batches = num_samples // num_w1_eval_samples
efp_jobs = None


# if running on nautilus
datasets_dir = "/graphganvol/MPGAN/datasets/"
evaluation_outputs_dir = "/graphganvol/MPGAN/ml4ps2023/evaluation_results/"
mp_trained_models = "/graphganvol/MPGAN/trained_models/"

# running locally
# datasets_dir = "./datasets/"
# evaluation_outputs_dir = "./ml4ps2023/evaluation_results/"
# mp_trained_models = "./trained_models/"


evaluation_outputs = evaluation_outputs_dir + "scores.json"
_ = os.system(f"mkdir -p {evaluation_outputs_dir}")


# mapping models to path to gen jets
model_name_map = {
    "g": {
        "mp": mp_trained_models + "mp_g/gen_jets.npy",
        "gapt_baseline": "/graphganvol/vk/g_128D_nh-8_n-2000/outputs/baseline2/best_epoch_gen_jets.npy",
        "gapt_gast": "/graphganvol/vk/g_128D_nh-8_n-2000/outputs/nz_condGD/best_epoch_gen_jets.npy",
        "gapt_igapt": "/graphganvol/vk/g_128D_nh-8_n-2000/outputs/nz_condGD_isab-nz2/best_epoch_gen_jets.npy",
    },
    "t": {
        "mp": mp_trained_models + "mp_t/gen_jets.npy",
        "gapt_baseline": "/graphganvol/vk/t_128D_nh-8_n-2000/outputs/baseline/best_epoch_gen_jets.npy",
        "gapt_gast": "/graphganvol/vk/t_128D_nh-8_n-2000/outputs/nz_condGD/best_epoch_gen_jets.npy",
        "gapt_igapt": "/graphganvol/vk/t_128D_nh-8_n-2000/outputs/nz_condGD_isab-nz/best_epoch_gen_jets.npy",
    },
    "q": {
        "mp": mp_trained_models + "mp_q/gen_jets.npy",
        "gapt_baseline": "/graphganvol/vk/q_128D_nh-8_n-2000/outputs/baseline/best_epoch_gen_jets.npy",
        "gapt_gast": "/graphganvol/vk/q_128D_nh-8_n-2000/outputs/nz_condGD/best_epoch_gen_jets.npy",
        "gapt_igapt": "/graphganvol/vk/q_128D_nh-8_n-2000/outputs/nz_condGD_isab-nz/best_epoch_gen_jets.npy",
    },
    "g150": {
        "gapt_baseline": "/graphganvol/anni/gapt/jets/mask/tune/m20/6000epochs/disnorm3x/test/best_epoch_gen_jets.npy",
        "gapt_gast": "/graphganvol/anni/gapt/jets/mask/tune/m20/6000epochs/cond/disnorm3x/test/best_epoch_gen_jets.npy",
        "gapt_igapt": "/graphganvol/anni/gapt/jets/mask/tune/m20/6000epochs/epicgan/condz/epic/layer48/test/best_epoch_gen_jets.npy",
    },
}

# evaluation scores
if os.path.exists(evaluation_outputs):
    with open(evaluation_outputs, "r") as f:
        scores = json.load(f)
else:
    scores = {}

datasets = ["g", "q", "t", "g150"]
# datasets = ["g", "q", "t"]
models = ["mp", "gapt_baseline", "gapt_gast", "gapt_igapt"]
# models = ["mp"]


def _save_scores(scores):
    """Save evaluation scores to json file"""
    with open(evaluation_outputs, "w") as f:
        json.dump(scores, f, indent=4)


def _check_load_efps(jets, path):
    """Check if EFPs have already been computed and load them if so"""
    if os.path.exists(path):
        efps = np.load(path)
    else:
        print(f"\t\tComputing EFPs and saving to {path}")
        efps = jetnet.utils.efps(jets, efpset_args=[("d<=", 4)], efp_jobs=efp_jobs)
        np.save(path, efps)

    return efps


def get_scores(jets1, jets2, efps1, efps2):
    """Compute evaluation scores between two sets of jets"""
    scores = {}

    scores["w1m"] = evaluation.w1m(
        jets1,
        jets2,
        num_eval_samples=num_w1_eval_samples,
        num_batches=num_w1_batches,
        return_std=True,
    )

    w1pm, w1pstd = evaluation.w1p(
        jets1,
        jets2,
        exclude_zeros=True,
        num_eval_samples=num_w1_eval_samples,
        num_batches=num_w1_batches,
        return_std=True,
    )

    scores["w1ppt"] = [w1pm[2], w1pstd[2]]
    scores["fpd"] = evaluation.fpd(efps1, efps2)
    scores["kpd"] = evaluation.kpd(efps1, efps2)

    return scores


# compute scores for real and generated jets
for jet_type in datasets:
    print(f"Evaluating {jet_type} jets")
    
    data_args = {
        "jet_type": jet_type,
        "data_dir": datasets_dir,
        "split_fraction": [0.7, 0.3, 0],
        "particle_features": ["etarel", "phirel", "ptrel"],
        "jet_features": None
    }
    
    if jet_type == "g150":
        data_args["num_particles"] = 150
        data_args["jet_type"] = "g"

    if jet_type not in scores:
        scores[jet_type] = {}

    data_args["split"] = "train"
    real_jets1, _ = JetNet.getData(**data_args)
    real_efps1 = _check_load_efps(real_jets1, f"{datasets_dir}/{jet_type}_efps1.npy")

    if "real" not in scores[jet_type]:
        print("\tEvaluating real jets")
        
        data_args["split"] = "valid"
        real_jets2, _ = JetNet.getData(**data_args)
        
        real_efps2 = _check_load_efps(real_jets2, f"{datasets_dir}/{jet_type}_efps2.npy")
        scores[jet_type]["real"] = get_scores(real_jets1, real_jets2, real_efps1, real_efps2)
        _save_scores(scores)

    for model in models:
        if model == "mp" and jet_type == "g150":
            continue
        
        print(f"\tEvaluating {model} jets")
        jet_path = model_name_map[jet_type][model]
        efp_path = jet_path.replace("jets.npy", "efps.npy")

        gen_jets = np.load(jet_path)[:num_samples, :, :3]
        gen_efps = _check_load_efps(gen_jets, efp_path)

        if model not in scores[jet_type]:
            scores[jet_type][model] = get_scores(real_jets1, gen_jets, real_efps1, gen_efps)
            _save_scores(scores)


def format_mean_sd(mean, sd):
    """round mean and standard deviation to most significant digit of sd and apply latex formatting"""
    decimals = -int(np.floor(np.log10(sd)))
    decimals -= int((sd * 10**decimals) >= 9.5)
    decimals = np.abs(decimals)
    if decimals < 0:
        ten_to = 10 ** (-decimals)
        if mean > ten_to:
            mean = ten_to * (mean // ten_to)
        else:
            if mean <= 0:
                print(f"Warning: mean is non-positive: {mean}")
                return "NaN"

            mean_ten_to = 10 ** np.floor(np.log10(mean))
            mean = mean_ten_to * (mean // mean_ten_to)
        sd = ten_to * (sd // ten_to)
        decimals = 0

    if mean >= 1e3 and sd >= 1e3:
        mean = np.round(mean * 1e-3)
        sd = np.round(sd * 1e-3)
        return f"${mean:.{decimals}f}$k $\\pm {sd:.{decimals}f}$k"
    else:
        return f"${mean:.{decimals}f} \\pm {sd:.{decimals}f}$"


def bold_best_key(val_str: str, bold: bool):
    if bold:
        return f"$\\mathbf{{{val_str[1:-1]}}}$"
    else:
        return val_str


# Make and save table

model_name_map = {
    "real": "Truth",
    "mp": "MPGAN",
    "gapt_baseline": "GAPT",
    "gapt_gast": "GAST",
    "gapt_igapt": "iGAPT",
}

jet_name_map = {"g": "Gluon (30)", "q": "Light quark (30)", "t": "Top quark (30)", "g150": "Gluon (150)"}

scores_scale_dict = {
    "fpd": 1e3,
    "kpd": 1e6,
    "w1m": 1e3,
    "w1ppt": 1e3,
}

row_order = ["real", "mp", "gapt_baseline", "gapt_gast", "gapt_igapt"]
# column_order = ["fpd", "kpd", "w1m", "w1ppt"]
column_order = ["w1ppt", "w1m", "fpd", "kpd"]

lines = []
for jet_type in datasets:
    if jet_type == "g150":
        row_order = ["real", "gapt_baseline", "gapt_gast", "gapt_igapt"]
    
    lines.append(rf"\multirow{{{len(row_order)}}}{{*}}{{{jet_name_map[jet_type]}}}" + "\n")
    
    # find the index of the lowest score for each metric
    bold_idxs = {}
    for col in column_order:
        bold_idxs[col] = np.argmin([scores[jet_type][key][col][0] for key in row_order[1:]]) + 1
    
    for i, key in enumerate(row_order):
        line = f" & {model_name_map[key]} & "
        
        format_scores = []
        for score in column_order:
            format_scores.append(bold_best_key(
                    format_mean_sd(
                        scores[jet_type][key][score][0] * scores_scale_dict[score],
                        scores[jet_type][key][score][1] * scores_scale_dict[score],
                    ),
                    i == bold_idxs[score],
                )
            )
        line += " & ".join(format_scores)
        
        line += "\\\\ \n"
        lines.append(line)


with open(f"{evaluation_outputs_dir}/scores_table.tex", "w") as f:
    f.writelines(lines)