import argparse

import torch
import numpy as np

from setup_training import models, get_model_args
from train import gen_multi_batch


feature_maxes = {
    "g": [1.4532885551452637, 0.520724892616272, 0.8537549376487732, 1.0],
    "q": [1.6211985349655151, 0.4568111002445221, 0.8896132111549377, 1.0],
    "t": [1.4242753982543945, 0.4949831962585449, 0.8774275183677673, 1.0],
}

feature_norms = [1.0, 1.0, 1.0, 1.0]
feature_shifts = [0.0, 0.0, -0.5, -0.5]


class objectview(object):
    """converts a dict into an object"""

    def __init__(self, d):
        self.__dict__ = d


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--G-state-dict",
        type=str,
        default="",
        help="Path to generator's state dict.",
    )

    parser.add_argument(
        "--G-args",
        type=str,
        default="",
        help="Path to generator's args file.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default="",
        help="# of samples to generate.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size when generating.",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="./gen_jets.npy",
        help="Path to gen jets output file.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Use CPU ('cpu') or GPU ('cuda') for generation.",
    )

    parser.add_argument(
        "--datasets-path",
        type=str,
        default="./datasets/",
        help="Path to gen jets output file.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"

    with open(args.G_args, "r") as f:
        G_args = objectview(eval(f.read()))

    G_args.device = args.device

    G = models(G_args, gen_only=True)
    G.load_state_dict(torch.load(args.G_state_dict, map_location=args.device))
    _, model_args, extra_args = get_model_args(G_args)

    if G_args.mask_c:
        from jetnet.datasets import JetNet

        labels = JetNet(
            G_args.jets, data_dir=args.datasets_path, train=False
        ).jet_features

        rng = np.random.default_rng()
        rand = rng.choice(len(labels), size=args.num_samples)
        labels = labels[rand].to(args.device)
    else:
        labels = None

    print("Generating samples")

    gen_jets = gen_multi_batch(
        model_args,
        G,
        args.batch_size,
        args.num_samples,
        G_args.num_hits,
        model=G_args.model,
        labels=labels,
        detach=True,
        **extra_args,
    )

    print("Generated samples")

    for i in range(3):
        if feature_shifts[i] is not None and feature_shifts[i] != 0:
            gen_jets[:, :, i] -= feature_shifts[i]

        if feature_norms[i] is not None:
            gen_jets[:, :, i] /= feature_norms[i]
            gen_jets[:, :, i] *= feature_maxes[G_args.jets][i]

    if G_args.mask:
        mask = gen_jets[:, :, -1] >= 0.5 if G_args.mask else None
        gen_jets[~mask] = 0

    gen_jets[:, :, 2][gen_jets[:, :, 2] < 0] = 0

    print("Unnormalized samples")

    np.save(args.output_file, gen_jets[:, :, :3])

    print("Saved samples")


if __name__ == "__main__":
    main()
