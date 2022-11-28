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
jets_dir = f"{mpgan_dir}/outputs/{args.name}/jets/"

_ = os.system(f"mkdir -p {jets_dir}")

kpd_path = f"{losses_dir}/kpd.txt"

feature_maxes = JetNet.fpnd_norm.feature_maxes
feature_maxes = feature_maxes + [1]

particle_norm = FeaturewiseLinearBounded(
    feature_norms=1.0,
    feature_shifts=[0.0, 0.0, -0.5, -0.5],
    feature_maxes=feature_maxes,
)

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
X = JetNet(**data_args, split="valid")
real_jf = X.jet_data

# load args
with open(f"{output_dir}/{args.name}_args.txt", "r") as f:
    model_args = setup_training.objectview(eval(f.read()))

G = setup_training.models(model_args, gen_only=True).to("cuda")

if os.path.exists(kpd_path):
    kpds = np.loadtxt(kpd_path)
    if kpds.ndim == 1:
        kpds = np.expand_dims(kpds, 0)

    kpds = list(kpds)
    start_idx = len(kpds) * 5
else:
    kpds = []
    start_idx = 0

for i in range(start_idx, 4001, 5):
    print(f"{datetime.datetime.now()} Starting Model {i}")

    G.load_state_dict(torch.load(f"{models_dir}/G_{i}.pt", map_location="cuda"))
    G.eval()

    if os.path.exists(f"{jets_dir}/{i}_gen_efps.npy"):
        gen_efps = np.load(f"{jets_dir}/{i}_gen_efps.npy")

    else:
        gen_jets = train.gen_multi_batch(
            {"embed_dim": model_args.gapt_embed_dim},
            G,
            2048,
            50000,
            30,
            model="gapt",
            out_device="cpu",
            labels=real_jf[:50000],
            detach=True,
        )

        gen_jets = jetnet.utils.gen_jet_corrections(
            particle_norm(gen_jets, inverse=True),
            ret_mask_separate=True,
            zero_mask_particles=True,
        )

        gen_mask = gen_jets[1]
        gen_jets = gen_jets[0]

        gen_mask = gen_mask.numpy()
        gen_jets = gen_jets.numpy()

        gen_efps = jetnet.utils.efps(gen_jets, efpset_args=[("d<=", 4)], efp_jobs=6)

        np.save(f"{jets_dir}/{i}_gen_jets.npy", gen_jets)
        np.save(f"{jets_dir}/{i}_gen_efps.npy", gen_efps)

    print(f"{datetime.datetime.now()} Calculating KPD")

    kpd = metrics.multi_batch_evaluation_mmd(real_efps, gen_efps)

    print(kpd)
    kpds.append(kpd)

    np.savetxt(kpd_path, kpds)
