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

name = "163_t_128D_nh8"
jets = "t"

epoch = 2400

args = parser.parse_args()

mpgan_dir = "/graphganvol/MPGAN/"
real_efps = np.load(f"{mpgan_dir}/efps/{jets}.npy")

output_dir = f"{mpgan_dir}/outputs/{name}/"
models_dir = f"{mpgan_dir}/outputs/{name}/models/"
losses_dir = f"{mpgan_dir}/outputs/{name}/losses/"

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
    "jet_type": jets,
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
with open(f"{output_dir}/{name}_args.txt", "r") as f:
    model_args = setup_training.objectview(eval(f.read()))

G = setup_training.models(model_args, gen_only=True).to("cuda")
G.load_state_dict(torch.load(f"{models_dir}/G_{epoch}.pt", map_location="cuda"))
G.eval()

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

gen_jets = gen_jets[0]
gen_jets = gen_jets.numpy()

np.save(f"{output_dir}/{epoch}_gen_jets.npy", gen_jets)
