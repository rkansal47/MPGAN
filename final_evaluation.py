import os

from jetnet.datasets import JetNet
from jetnet import evaluation

import numpy as np


datasets = ["g", "q", "t"]
samples_dict = {key: {} for key in datasets}
real_samples = {}
num_samples = 50000

# mapping folder names to gen and disc names in the final table
model_name_map = {
    "fc": ["FC", "FC"],
    "fcmp": ["FC", "MP"],
    "fcpnet": ["FC", "PointNet"],
    "graphcnn": ["GraphCNN", "FC"],
    "graphcnnmp": ["GraphCNN", "MP"],
    "graphcnnpnet": ["GraphCNN", "PointNet"],
    "mp": ["MP", "MP"],
    "mpfc": ["MP", "FC"],
    "mplfc": ["MP-LFC", "MP"],
    "mppnet": ["MP", "PointNet"],
    "treeganfc": ["TreeGAN", "FC"],
    "treeganmp": ["TreeGAN", "MP"],
    "treeganpnet": ["TreeGAN", "PointNet"],
}


# Load samples

# models_dir = "/graphganvol/MPGAN/trained_models/"
models_dir = "./trained_models/"

for dir in os.listdir(models_dir):
    if dir == ".DS_Store" or dir == "README.md":
        continue

    model_name = dir.split("_")[0]

    if model_name in model_name_map:
        dataset = dir.split("_")[1]
        samples = np.load(f"{models_dir}/{dir}/gen_jets.npy")[:num_samples, :, :3]
        samples_dict[dataset][model_name] = samples

for dataset in datasets:
    real_samples[dataset] = (
        JetNet(
            dataset, "/graphganvol/MPGAN/datasets/", normalize=False, train=False, use_mask=False
        )
        .data[:num_samples]
        .numpy()
    )

# order in final table
order = [
    "fc",
    "graphcnn",
    "treeganfc",
    "fcpnet",
    "graphcnnpnet",
    "treeganpnet",
    "-",
    "mp",
    "mplfc",
    "-",
    "fcmp",
    "graphcnnmp",
    "treeganmp",
    "mpfc",
    "mppnet",
]


# get evaluation metrics for all samples and save in folder
evals_dict = {key: {} for key in datasets}

for key in order:
    print(key)
    for dataset in datasets:
        print(dataset)
        if key not in samples_dict[dataset]:
            print(f"{key} samples for {dataset} jets not found")
            continue

        gen_jets = samples_dict[dataset][key]
        real_jets = real_samples[dataset]

        evals = {}

        evals["w1m"] = evaluation.w1m(gen_jets, real_jets)
        evals["w1p"] = evaluation.w1p(gen_jets, real_jets, average_over_features=False)
        evals["w1efp"] = evaluation.w1efp(gen_jets, real_jets, average_over_efps=False)
        evals["fpnd"] = evaluation.fpnd(gen_jets[:, :30], dataset, device="cuda", batch_size=256)
        cov, mmd = evaluation.cov_mmd(real_jets, gen_jets)
        evals["coverage"] = cov
        evals["mmd"] = mmd

        f = open(f"{models_dir}/{key}_{dataset}/evals.txt", "w+")
        f.write(str(evals))
        f.close()

        evals_dict[dataset][key] = evals


# load eval metrics if already saved
from numpy import array

for dataset in datasets:
    for key in order:
        if key != "-":
            with open(f"{models_dir}/{key}_{dataset}/evals.txt", "r") as f:
                evals_dict[dataset][key] = eval(f.read())

            if "cov" in evals_dict[dataset][key]:
                evals_dict[dataset][key]["coverage"] = evals_dict[dataset][key]["cov"]
                del evals_dict[dataset][key]["cov"]

                with open(f"{models_dir}/{key}_{dataset}/evals.txt", "w") as f:
                    f.write(str(evals_dict[dataset][key]))


# find best values

best_key_dict = {key: {} for key in datasets}
eval_keys = ["w1m", "w1p", "w1efp", "fpnd", "coverage", "mmd"]

for dataset in datasets:
    model_keys = list(evals_dict[dataset].keys())

    lists = {key: [] for key in eval_keys}

    for key in model_keys:
        evals = evals_dict[dataset][key]

        lists["w1m"].append(np.round(evals["w1m"][0], 5))
        lists["w1p"].append(np.round(np.mean(evals["w1p"][0]), 4))
        lists["w1efp"].append(np.round(np.mean(evals["w1efp"][0]), 5 if dataset == "t" else 6))
        lists["fpnd"].append(np.round(evals["fpnd"], 2))
        lists["coverage"].append(1 - np.round(evals["coverage"], 2))  # invert to maximize cov
        lists["mmd"].append(np.round(evals["mmd"], 3))

    for key in eval_keys:
        best_key_dict[dataset][key] = np.array(model_keys)[
            np.flatnonzero(np.array(lists[key]) == np.array(lists[key]).min())
        ]


def format_mean_sd(mean, sd):
    """round mean and standard deviation to most significant digit of sd and apply latex formatting"""
    decimals = -int(np.floor(np.log10(sd)))
    decimals -= int((sd * 10**decimals) >= 9.5)

    if decimals < 0:
        ten_to = 10 ** (-decimals)
        if mean > ten_to:
            mean = ten_to * (mean // ten_to)
        else:
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


def format_fpnd(fpnd):
    if fpnd >= 1e6:
        fpnd = np.round(fpnd * 1e-6)
        return f"${fpnd:.0f}$M"
    elif fpnd >= 1e3:
        fpnd = np.round(fpnd * 1e-3)
        return f"${fpnd:.0f}$k"
    elif fpnd >= 10:
        fpnd = np.round(fpnd)
        return f"${fpnd:.0f}$"
    elif fpnd >= 1:
        return f"${fpnd:.1f}$"
    else:
        return f"${fpnd:.2f}$"


def bold_best_key(val_str: str, bold: bool):
    if bold:
        return f"$\\mathbf{{{val_str[1:-1]}}}$"
    else:
        return val_str


# Make and save table

table_dict = {key: {} for key in datasets}

for dataset in datasets:
    lines = []
    for key in order:
        if key == "-":
            lines.append("\cmidrule(lr){2-3}\n")
        else:
            line = f" & {model_name_map[key][0]} & {model_name_map[key][1]}"
            evals = evals_dict[dataset][key]

            line += " & " + bold_best_key(
                format_mean_sd(evals["w1m"][0] * 1e3, evals["w1m"][1] * 1e3),
                key in best_key_dict[dataset]["w1m"],
            )

            line += " & " + bold_best_key(
                format_mean_sd(
                    np.mean(evals["w1p"][0]) * 1e3, np.linalg.norm(evals["w1p"][1]) * 1e3
                ),
                key in best_key_dict[dataset]["w1p"],
            )

            line += " & " + bold_best_key(
                format_mean_sd(
                    np.mean(evals["w1efp"][0]) * 1e5, np.linalg.norm(evals["w1efp"][1]) * 1e5
                ),
                key in best_key_dict[dataset]["w1efp"],
            )

            line += " & " + bold_best_key(
                format_fpnd(evals["fpnd"]), key in best_key_dict[dataset]["fpnd"]
            )

            line += " & " + bold_best_key(
                f"${evals['coverage']:.2f}$", key in best_key_dict[dataset]["coverage"]
            )

            line += " & " + bold_best_key(
                f"${evals['mmd']:.3f}$", key in best_key_dict[dataset]["mmd"]
            )

            line += "\\\\ \n"

            lines.append(line)

    table_dict[dataset] = lines

    with open(f"evaluation_results/{dataset}.tex", "w") as f:
        f.writelines(table_dict[dataset])
