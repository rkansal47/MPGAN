import os
from copy import copy
from tqdm import tqdm

from jetnet import utils
from jetnet.datasets import JetNet

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


datasets = ["g", "q", "t"]
samples_dict = {key: {} for key in datasets}
num_samples = 50000

model_name_map = {
    "fcpnet": "FC",
    "graphcnnpnet": "GraphCNN",
    "treeganpnet": "TreeGAN",
    "mp": "MP",
    # 'mppnet': 'MPPNet'
    # "pcgan": "PCGAN"
}

models_dir = "trained_models/"
plot_dir = "plots/"

for dir in os.listdir(models_dir):
    if dir == ".DS_Store" or dir == "README.md":
        continue

    model_name = dir.split("_")[0]

    if model_name in model_name_map:
        dataset = dir.split("_")[1]
        samples = np.load(f"{models_dir}/{dir}/gen_jets.npy")[:num_samples, :, :3]
        samples_dict[dataset][model_name_map[model_name]] = samples

for dataset in samples_dict:
    samples_dict[dataset]["Real"] = (
        JetNet(dataset, "./datasets/", normalize=False, train=False, use_mask=False)
        .data[:num_samples]
        .numpy()
    )

line_opts = {
    "Real": {"color": "red", "linewidth": 3, "linestyle": "solid"},
    "MP": {"color": "blue", "linewidth": 3, "linestyle": "dashed"},
    "FC": {"color": "green", "linewidth": 3, "linestyle": "dashdot"},
    "GraphCNN": {"color": "brown", "linewidth": 3, "linestyle": "dashed"},
    "TreeGAN": {"color": "orange", "linewidth": 3, "linestyle": "dashdot"},
    # "PCGAN": {"color": "purple", "linewidth": 3, "linestyle": "dashed"},
    # 'MPPNET': {'color': 'purple', 'linewidth': 2, 'linestyle': (0, (5, 10))},
}

print("Getting EFPs")
efps = {}
for dataset in samples_dict:
    efps[dataset] = {}
    for key in line_opts:
        print(key)
        efps[dataset][key] = utils.efps(samples_dict[dataset][key])[:, 0]

plt.rcParams.update({"font.size": 28})
plt.style.use(hep.style.CMS)

print("Plotting feature distributions")
fig = plt.figure(figsize=(36, 24))
for i in range(len(datasets)):
    dataset = datasets[i]
    if dataset == "g":
        efpbins = np.linspace(0, 0.0013, 51)
        pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.1, 101)]
        ylims = [2e5, 1e5, 1e4, 3e4]
    elif dataset == "q":
        efpbins = np.linspace(0, 0.002, 51)
        pbins = [np.linspace(-0.3, 0.3, 101), np.linspace(0, 0.15, 101)]
        ylims = [1e5, 0.3e6, 0.125e5, 0.3e5]
    elif dataset == "t":
        efpbins = np.linspace(0, 0.0045, 51)
        pbins = [np.arange(-0.5, 0.5, 0.005), np.arange(0, 0.1, 0.001)]
        ylims = [1e5, 1e5, 0.5e4, 0.6e4]

    mbins = np.linspace(0, 0.225, 51)

    fig.add_subplot(3, 4, i * 4 + 1)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    plt.xlabel("Particle $\eta^{rel}$")
    plt.ylabel("Number of Particles")

    for key in line_opts.keys():
        samples = samples_dict[dataset][key]

        # remove zero-padded particles
        mask = np.linalg.norm(samples, axis=2) != 0
        parts = samples[mask]

        _ = plt.hist(parts[:, 0], pbins[0], histtype="step", label=key, **line_opts[key])

    plt.legend(loc=1, prop={"size": 18}, fancybox=True)
    plt.ylim(0, ylims[0])

    fig.add_subplot(3, 4, i * 4 + 2)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    plt.xlabel("Particle $p_T^{rel}$")
    plt.ylabel("Number of Particles")

    for key in line_opts.keys():
        samples = samples_dict[dataset][key]

        # remove zero-padded particles
        mask = np.linalg.norm(samples, axis=2) != 0
        parts = samples[mask]

        _ = plt.hist(parts[:, 2], pbins[1], histtype="step", label=key, **line_opts[key])

    plt.legend(loc=1, prop={"size": 18}, fancybox=True)
    plt.ylim(0, ylims[1])

    fig.add_subplot(3, 4, i * 4 + 3)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    plt.xlabel("Relative Jet Mass")
    plt.ylabel("Number of Jets")

    for key in line_opts.keys():
        masses = utils.jet_features(samples_dict[dataset][key])["mass"]
        _ = plt.hist(masses, mbins, histtype="step", label=key, **line_opts[key])

    plt.legend(loc=1, prop={"size": 18}, fancybox=True)
    plt.ylim(0, ylims[2])

    fig.add_subplot(3, 4, i * 4 + 4)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    plt.ticklabel_format(axis="x", scilimits=(0, 0), useMathText=True)
    plt.xlabel("Jet EFP", x=0.7)
    plt.ylabel("Number of Jets")

    for key in line_opts.keys():
        _ = plt.hist(efps[dataset][key], efpbins, histtype="step", label=key, **line_opts[key])

    plt.legend(loc=1, prop={"size": 18}, fancybox=True)
    plt.ylim(0, ylims[3])


plt.tight_layout(pad=0.5)
plt.savefig(f"{plot_dir}/pcgan_feature_distributions.pdf", bbox_inches="tight")
plt.show()

print("Plotting jet images")
average_images = {key: {} for key in datasets}

# from mpl_toolkits.axes_grid1 import make_axes_locatable


order = ["Real", "MP", "FC", "GraphCNN", "TreeGAN"]  # , "PCGAN"]

num_images = 4
np.random.seed(2021)
rand_sample = np.random.randint(num_samples, size=num_images)

im_size = 25
maxR = 0.4
ave_maxR = 0.5

cm = copy(plt.cm.jet)
cm.set_under(color="white")

plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)

for dataset in datasets:
    fig, axes = plt.subplots(
        nrows=len(order),
        ncols=num_images + 1,
        figsize=(40, 48),
        gridspec_kw={"wspace": 0.25, "hspace": 0},
    )

    for j in range(len(order)):
        key = order[j]
        axes[j][0].annotate(
            key,
            xy=(0, -1),
            xytext=(-axes[j][0].yaxis.labelpad - 15, 0),
            xycoords=axes[j][0].yaxis.label,
            textcoords="offset points",
            ha="right",
            va="center",
        )

        samples = samples_dict[dataset][key]
        rand_samples = samples[rand_sample]

        for i in range(num_images):
            im = axes[j][i].imshow(
                utils.to_image(rand_samples[i], im_size, maxR=maxR),
                cmap=cm,
                interpolation="nearest",
                vmin=1e-8,
                extent=[-maxR, maxR, -maxR, maxR],
                vmax=0.05,
            )
            axes[j][i].tick_params(which="both", bottom=False, top=False, left=False, right=False)
            axes[j][i].set_xlabel("$\phi^{rel}$")
            axes[j][i].set_ylabel("$\eta^{rel}$")

        # average jet image

        if key not in average_images[dataset]:
            ave_im = np.zeros((im_size, im_size))
            for i in tqdm(range(10000)):
                ave_im += utils.to_image(samples[i], im_size, maxR=ave_maxR)
            ave_im /= 10000
            average_images[dataset][key] = ave_im

        im = axes[j][-1].imshow(
            average_images[dataset][key],
            cmap=plt.cm.jet,
            interpolation="nearest",
            vmin=1e-8,
            extent=[-ave_maxR, ave_maxR, -ave_maxR, ave_maxR],
            vmax=0.05,
        )
        axes[j][-1].set_title("Average Jet Image", pad=5)
        axes[j][-1].tick_params(which="both", bottom=False, top=False, left=False, right=False)
        axes[j][-1].set_xlabel("$\phi^{rel}$")
        axes[j][-1].set_ylabel("$\eta^{rel}$")

        cbar = fig.colorbar(im, ax=axes[j].ravel().tolist(), fraction=0.007)
        cbar.set_label("$p_T^{rel}$")

    fig.tight_layout()
    plt.savefig(f"{plot_dir}/jet_images_{dataset}.pdf", bbox_inches="tight")
    plt.show()
