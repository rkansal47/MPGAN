import os

from jetnet.datasets import JetNet
from jetnet import evaluation

import numpy as np


datasets = ["g", "q", "t"]
samples_dict = {key: {} for key in datasets}
real_samples = {}
num_samples = 50000

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

models_dir = "/graphganvol/MPGAN/trained_models/"

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
        evals["fpnd"] = evaluation.fpnd(gen_jets, dataset, device="cuda", batch_size=256)
        cov, mmd = evaluation.cov_mmd(real_jets, gen_jets)
        evals["cov"] = cov
        evals["mmd"] = mmd

        f = open(f"{models_dir}/{key}_{dataset}/evals.txt", "w+")
        f.write(str(evals))
        f.close()

        evals_dict[dataset][key] = evals
