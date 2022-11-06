import argparse
import torch
import numpy as np

import train, setup_training
import metrics
import jetnet
from jetnet.datasets import JetNet
from jetnet.datasets.normalisations import FeaturewiseLinearBounded, FeaturewiseLinear

import datetime

import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
    type=str,
    default="test",
    help="name or tag for model; will be appended with other info",
)

parser.add_argument(
    "--jets",
    type=str,
    default="g",
    help="jet type",
    choices=["g", "t", "w", "z", "q"],
)

args = parser.parse_args()

mpgan_dir = "/graphganvol/MPGAN/"
real_efps = np.load(f"{mpgan_dir}/efps/{args.jets}.npy")

output_dir = f"{mpgan_dir}/outputs/{args.name}/"
models_dir = f"{mpgan_dir}/outputs/{args.name}/models/"
losses_dir = f"{mpgan_dir}/outputs/{args.name}/losses/"

kpd_path = f"{losses_dir}/kpd.txt"

jet_norm = FeaturewiseLinear(feature_scales=1.0 / 30)
data_args = {
    "jet_type": args.jets,
    "data_dir": f"{mpgan_dir}/datasets/",
    "num_particles": 30,
    "particle_features": None,
    "jet_features": "num_particles",
    "jet_normalisation": jet_norm,
    "split_fraction": [0.7, 0.3, 0],
}
real_jf = JetNet.getData(**data_args, split="valid")

# load args
with open(f"{output_dir}/{args.name}_args.txt", "r") as f:
    model_args = setup_training.objectview(eval(f.read()))

G = setup_training.models(model_args, gen_only=True)

if os.path.exists(kpd_path):
    kpds = list(np.loadtxt(kpd_path))
    start_idx = len(kpds)
else:
    kpds = []
    start_idx = 0

for i in range(start_idx, 4001, 5):
    print(f"{datetime.datetime.now()} Starting Model {i}")

    G.load_state_dict(torch.load(f"{models_dir}/G_{i}.pt", map_location="cuda"))
    G.eval()

    gen_jets = train.gen_multi_batch(
        {}, G, 1024, 50000, 30, model="gapt", out_device="cpu", label=real_jf[:50000], detach=True
    )
    gen_efps = jetnet.utils.efps(gen_jets, efpset_args=[("d<=", 4)], efp_jobs=6)

    print(f"{datetime.datetime.now()} Calculating KPD")

    kpd = metrics.multi_batch_evaluation(
        real_efps, gen_efps, 10, 5000, metrics.mmd_poly_quadratic_unbiased
    )

    print(kpd)
    kpds.append(kpd)

    np.savetxt(kpd_path, kpds)
