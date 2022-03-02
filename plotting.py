import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import figure
import mplhep as hep
from matplotlib.colors import LogNorm
import gc

# matplotlib.use("PDF")
# plt.switch_backend("agg")
plt.rcParams.update({"font.size": 16})
plt.style.use(hep.style.CMS)


LAYER_SPECS = [(3, 96), (12, 12), (12, 6)]

from guppy import hpy

h = hpy()


def plot_hit_feats(
    real_jets,
    gen_jets,
    real_mask=None,
    gen_mask=None,
    coords="polarrel",
    name=None,
    figs_path=None,
    dataset="jetnet",
    num_particles=30,
    losses=None,
    const_ylim=False,
    show=False,
):
    """Plot particle feature histograms"""
    plabels = ["$\eta$", "$\phi$", "z", "E (GeV)"]
    pbins = [
        np.linspace(0, 1, 51),
        np.linspace(0, 1, 51),
        np.linspace(0, 1, 51),
        np.linspace(0, 50000, 51),
    ]

    if real_mask is not None:
        parts_real = real_jets[real_mask]
        parts_gen = gen_jets[gen_mask]
    else:
        parts_real = real_jets.reshape(-1, real_jets.shape[2])
        parts_gen = gen_jets.reshape(-1, gen_jets.shape[2])

    print("pre plot")
    print(h.heap())

    # plt.switch_backend("agg")

    fig = plt.figure(figsize=(16, 16))

    for i in range(4):
        fig.add_subplot(2, 2, i + 1)
        plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
        _ = plt.hist(parts_real[:, i], pbins[i], histtype="step", label="Real", color="red")
        _ = plt.hist(parts_gen[:, i], pbins[i], histtype="step", label="Generated", color="blue")
        plt.xlabel("Hit " + plabels[i])
        plt.ylabel("Number of Hits")
        if i == 3:
            plt.yscale("log")
        plt.legend(loc=1, prop={"size": 18})

    plt.tight_layout(pad=2.0)
    if figs_path is not None and name is not None:
        plt.savefig(figs_path + name + ".pdf", bbox_inches="tight")

    print("before close")
    print(h.heap())

    if show:
        plt.show()
    else:
        plt.close()
        gc.collect()

        print("after close")
        print(h.heap())


def plot_layerwise_hit_feats(
    real_showers,
    gen_showers,
    real_mask=None,
    gen_mask=None,
    name=None,
    figs_path=None,
    num_particles=30,
    show=False,
):
    real_hits = real_showers[real_mask]
    gen_hits = gen_showers[gen_mask]

    real_layer_etas = []
    real_layer_phis = []
    real_layer_Es = []

    gen_layer_etas = []
    gen_layer_phis = []
    gen_layer_Es = []

    for i in range(3):
        real_layer_hits = real_hits[
            (real_hits[:, 2] > i * 0.33) * (real_hits[:, 2] < (i + 1) * 0.33)
        ]
        real_layer_etas.append(((real_layer_hits[:, 0] * LAYER_SPECS[i][1]) - 0.5).astype(int))
        real_layer_phis.append(((real_layer_hits[:, 1] * LAYER_SPECS[i][0]) - 0.5).astype(int))
        real_layer_Es.append(real_layer_hits[:, 3])

        gen_layer_hits = gen_hits[(gen_hits[:, 2] > i * 0.33) * (gen_hits[:, 2] < (i + 1) * 0.33)]
        gen_layer_etas.append(((gen_layer_hits[:, 0] * LAYER_SPECS[i][1]) - 0.5).astype(int))
        gen_layer_phis.append(((gen_layer_hits[:, 1] * LAYER_SPECS[i][0]) - 0.5).astype(int))
        gen_layer_Es.append(gen_layer_hits[:, 3])

    fig = plt.figure(figsize=(22, 22), constrained_layout=True)
    fig.suptitle(" ")

    subfigs = fig.subfigures(nrows=3, ncols=1)

    for i, subfig in enumerate(subfigs):
        subfig.suptitle(f"Layer {i + 1}")

        # create 1x3 subplots per subfig
        axs = subfig.subplots(nrows=1, ncols=3)
        axs[0].hist(
            real_layer_etas[i], bins=np.arange(LAYER_SPECS[i][1] + 1), histtype="step", label="Real"
        )
        axs[0].hist(
            gen_layer_etas[i],
            bins=np.arange(LAYER_SPECS[i][1] + 1),
            histtype="step",
            label="Generated",
        )
        axs[0].ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
        axs[0].set_xlabel(r"Hit $\eta$s")
        axs[0].set_ylabel("Number of Hits")
        axs[0].legend()

        axs[1].hist(
            real_layer_phis[i], bins=np.arange(LAYER_SPECS[i][0] + 1), histtype="step", label="Real"
        )
        axs[1].hist(
            gen_layer_phis[i],
            bins=np.arange(LAYER_SPECS[i][0] + 1),
            histtype="step",
            label="Generated",
        )
        axs[1].ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
        axs[1].set_xlabel(r"Hit $\varphi$s")
        axs[1].set_ylabel("Number of Hits")
        axs[1].legend()

        bins = np.linspace(0, 50, 51) if i < 2 else np.linspace(0, 4, 51)
        axs[2].hist(real_layer_Es[i] / 1000, bins=bins, histtype="step", label="Real")
        axs[2].hist(gen_layer_Es[i] / 1000, bins=bins, histtype="step", label="Generated")
        axs[2].set_xlabel("Hit Energies (GeV)")
        axs[2].set_ylabel("Number of Hits")
        axs[2].set_yscale("log")
        axs[2].legend()

    # plt.tight_layout(2.0)
    if figs_path is not None and name is not None:
        print(f"saving fig at {figs_path} {name}")
        plt.savefig(figs_path + name + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def to_image(phi_size, eta_size, phis, etas, Es):
    jet_image = np.zeros((phi_size, eta_size))

    for eta, phi, pt in zip(phis, etas, Es):
        if eta >= 0 and eta < eta_size and phi >= 0 and phi < phi_size:
            jet_image[phi, eta] += pt

    return jet_image


def plot_shower_ims(
    gen_showers,
    name=None,
    figs_path=None,
    show=False,
):
    gen_ims = []

    NUM_IMS = 1000

    for j in range(3):
        gen_layer_ims = []
        for i in range(NUM_IMS):
            gen_layer_hits = gen_showers[i][
                (gen_showers[i][:, 2] > j * 0.33) * (gen_showers[i][:, 2] < (j + 1) * 0.33)
            ]
            gen_etas = ((gen_layer_hits[:, 0] * LAYER_SPECS[j][1]) - 0.5).astype(int)
            gen_phis = ((gen_layer_hits[:, 1] * LAYER_SPECS[j][0]) - 0.5).astype(int)
            gen_Es = gen_layer_hits[:, 3]
            gen_layer_ims.append(
                to_image(LAYER_SPECS[j][0], LAYER_SPECS[j][1], gen_etas, gen_phis, gen_Es)
            )

        gen_ims.append(gen_layer_ims)

    vmins = [1e-2, 1e-2, 1e-4]
    vmaxs = [1e5, 1e3, 10]

    fig = plt.figure(figsize=(24, 32), constrained_layout=True)
    fig.suptitle(" ")

    subfigs = fig.subfigures(nrows=1, ncols=3)

    for i, subfig in enumerate(subfigs):
        subfig.suptitle(f"Layer {i + 1}")

        axs = subfig.subplots(nrows=6, ncols=1)

        for j in range(5):
            pos = axs[j].imshow(
                gen_ims[i][j],
                aspect="auto",
                cmap="binary",
                vmin=vmins[i],
                vmax=vmaxs[i],
                norm=LogNorm(),
                extent=(0, LAYER_SPECS[i][1], 0, LAYER_SPECS[i][0]),
            )
            axs[j].set_xlabel(r"$\eta$")
            axs[j].set_ylabel(r"$\varphi$")
            fig.colorbar(pos, ax=axs[j])

        j = 5
        axs[j].set_title("Average Image")
        pos = axs[j].imshow(
            np.mean(np.array(gen_ims[i]), axis=0),
            aspect="auto",
            cmap="binary",
            vmin=vmins[i],
            vmax=vmaxs[i],
            norm=LogNorm(),
            extent=(0, LAYER_SPECS[i][1], 0, LAYER_SPECS[i][0]),
        )
        axs[j].set_xlabel(r"$\eta$")
        axs[j].set_ylabel(r"$\varphi$")
        fig.colorbar(pos, ax=axs[j])

    if figs_path is not None and name is not None:
        plt.savefig(figs_path + name + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_part_feats_jet_mass(
    jet_type,
    real_jets,
    gen_jets,
    real_mask,
    gen_mask,
    real_masses,
    gen_masses,
    num_particles=30,
    coords="polarrel",
    name=None,
    figs_path=None,
    dataset="jetnet",
    losses=None,
    const_ylim=False,
    show=False,
):
    """Plot histograms of particle feature + jet mass in one row"""
    if coords == "cartesian":
        plabels = ["$p_x$ (GeV)", "$p_y$ (GeV)", "$p_z$ (GeV)"]
        bin = np.arange(-500, 500, 10)
        pbins = [bin, bin, bin]
    elif coords == "polarrel":
        plabels = ["$\eta^{rel}$", "$\phi^{rel}$", "$p_T^{rel}$"]
        if jet_type == "g" or jet_type == "q" or jet_type == "w" or jet_type == "z":
            if num_particles == 100:
                pbins = [
                    np.arange(-0.5, 0.5, 0.005),
                    np.arange(-0.5, 0.5, 0.005),
                    np.arange(0, 0.1, 0.001),
                ]
            else:
                pbins = [
                    np.linspace(-0.3, 0.3, 100),
                    np.linspace(-0.3, 0.3, 100),
                    np.linspace(0, 0.2, 100),
                ]
        elif jet_type == "t":
            pbins = [
                np.linspace(-0.5, 0.5, 100),
                np.linspace(-0.5, 0.5, 100),
                np.linspace(0, 0.2, 100),
            ]
    elif coords == "polarrelabspt":
        plabels = ["$\eta^{rel}$", "$\phi^{rel}$", "$p_T (GeV)$"]
        pbins = [np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01), np.arange(0, 400, 4)]

    if jet_type == "g" or jet_type == "q" or jet_type == "t":
        mbins = np.linspace(0, 0.225, 51)
    else:
        mbins = np.linspace(0, 0.12, 51)

    if real_mask is not None:
        parts_real = real_jets[real_mask]
        parts_gen = gen_jets[gen_mask]
    else:
        parts_real = real_jets.reshape(-1, real_jets.shape[2])
        parts_gen = gen_jets.reshape(-1, gen_jets.shape[2])

    fig = plt.figure(figsize=(30, 8))

    for i in range(3):
        fig.add_subplot(1, 4, i + 1)
        plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
        _ = plt.hist(parts_real[:, i], pbins[i], histtype="step", label="Real", color="red")
        _ = plt.hist(parts_gen[:, i], pbins[i], histtype="step", label="Generated", color="blue")
        plt.xlabel("Particle " + plabels[i])
        plt.ylabel("Number of Particles")
        if losses is not None and "w1p" in losses:
            plt.title(
                f'$W_1$ = {losses["w1p"][-1][i]:.2e} ± {losses["w1p"][-1][i + len(losses["w1p"][-1]) // 2]:.2e}',
                fontsize=12,
            )

        plt.legend(loc=1, prop={"size": 18})

    fig.add_subplot(1, 4, 4)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    _ = plt.hist(real_masses, bins=mbins, histtype="step", label="Real", color="red")
    _ = plt.hist(gen_masses, bins=mbins, histtype="step", label="Generated", color="blue")
    plt.xlabel("Jet $m/p_{T}$")
    plt.ylabel("Jets")
    plt.legend(loc=1, prop={"size": 18})
    if losses is not None and "w1m" in losses:
        plt.title(f'$W_1$ = {losses["w1m"][-1][0]:.2e} ± {losses["w1m"][-1][1]:.2e}', fontsize=12)

    plt.tight_layout(2.0)
    if figs_path is not None and name is not None:
        plt.savefig(figs_path + name + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_jet_feats(
    jet_type,
    real_masses,
    gen_masses,
    real_efps,
    gen_efps,
    name=None,
    figs_path=None,
    losses=None,
    show=False,
):
    """Plot 5 EFPs and jet mass histograms"""
    if jet_type == "g":
        binranges = [0.0013, 0.0004, 0.0004, 0.0004, 0.0004]
    elif jet_type == "q":
        binranges = [0.002, 0.001, 0.001, 0.0005, 0.0005]
    else:
        binranges = [0.0045, 0.0035, 0.004, 0.002, 0.003]

    bins = [np.linspace(0, binr, 101) for binr in binranges]

    if jet_type == "g" or jet_type == "q" or jet_type == "t":
        mbins = np.linspace(0, 0.225, 51)
    else:
        mbins = np.linspace(0, 0.12, 51)

    fig = plt.figure(figsize=(20, 12))

    fig.add_subplot(2, 3, 1)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    _ = plt.hist(real_masses, bins=mbins, histtype="step", label="Real", color="red")
    _ = plt.hist(gen_masses, bins=mbins, histtype="step", label="Generated", color="blue")
    plt.xlabel("Jet $m/p_{T}$")
    plt.ylabel("Jets")
    plt.legend(loc=1, prop={"size": 18})
    if losses is not None and "w1m" in losses:
        plt.title(f'$W_1$ = {losses["w1m"][-1][0]:.2e} ± {losses["w1m"][-1][1]:.2e}', fontsize=12)

    for i in range(5):
        fig.add_subplot(2, 3, i + 2)
        plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
        plt.ticklabel_format(axis="x", scilimits=(0, 0), useMathText=True)
        _ = plt.hist(real_efps[:, i], bins[i], histtype="step", label="Real", color="red")
        _ = plt.hist(gen_efps[:, i], bins[i], histtype="step", label="Generated", color="blue")
        plt.xlabel("EFP " + str(i + 1), x=0.7)
        plt.ylabel("Jets")
        plt.legend(loc=1, prop={"size": 18})
        if losses is not None and "w1efp" in losses:
            plt.title(
                f'$W_1$ = {losses["w1efp"][-1][i]:.2e} ± {losses["w1efp"][-1][i + len(losses["w1efp"][-1]) // 2]:.2e}',
                fontsize=12,
            )

    plt.tight_layout(pad=0.5)
    if figs_path is not None and name is not None:
        plt.savefig(figs_path + name + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


# def plot_jet_mass_pt(realjf, genjf, dataset="jetnet", name=None, figs_path=None, show=False):
#     if dataset == "jetnet":
#         jlabels = ["Jet Relative Mass", "Jet Relative $p_T$"]
#         binsm = np.linspace(0, 0.225, 101)
#         binspt = np.linspace(0.5, 1.2, 101)
#     elif dataset == "jets-lagan":
#         jlabels = ["Jet Mass (GeV)", "Jet $p_T$ (GeV)"]
#         binsm = np.linspace(40, 120, 51)
#         binspt = np.linspace(220, 340, 51)
#
#     fig = plt.figure(figsize=(16, 8))
#
#     fig.add_subplot(1, 2, 1)
#     plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
#     # plt.ticklabel_format(axis='x', scilimits=(0, 0), useMathText=True)
#     _ = plt.hist(realjf[:, 0], bins=binsm, histtype="step", label="Real", color="red")
#     _ = plt.hist(genjf[:, 0], bins=binsm, histtype="step", label="Generated", color="blue")
#     plt.xlabel(jlabels[0])
#     plt.ylabel("Jets")
#     plt.legend(loc=1, prop={"size": 18})
#
#     fig.add_subplot(1, 2, 2)
#     plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
#     plt.ticklabel_format(axis="x", scilimits=(0, 0), useMathText=True)
#     _ = plt.hist(realjf[:, 1], bins=binspt, histtype="step", label="Real", color="red")
#     _ = plt.hist(genjf[:, 1], bins=binspt, histtype="step", label="Generated", color="blue")
#     plt.xlabel(jlabels[1])
#     plt.ylabel("Jets")
#     plt.legend(loc=1, prop={"size": 18})
#
#     plt.tight_layout(pad=2)
#     if figs_path is not None and name is not None:
#         plt.savefig(figs_path + name + ".pdf", bbox_inches="tight")
#
#     if show:
#         plt.show()
#     else:
#         plt.close()


def plot_losses(losses, loss="lg", name=None, losses_path=None, show=False):
    """Plot loss curves"""
    plt.figure()

    if loss == "og" or loss == "ls":
        plt.plot(losses["Dr"], label="Discriminitive real loss")
        plt.plot(losses["Df"], label="Discriminitive fake loss")
        plt.plot(losses["G"], label="Generative loss")
    elif loss == "w":
        plt.plot(losses["D"], label="Critic loss")
    elif loss == "hinge":
        plt.plot(losses["Dr"], label="Discriminitive real loss")
        plt.plot(losses["Df"], label="Discriminitive fake loss")
        plt.plot(losses["G"], label="Generative loss")

    if "gp" in losses:
        plt.plot(losses["gp"], label="Gradient penalty")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if losses_path is not None and name is not None:
        plt.savefig(losses_path + name + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_eval(
    losses, epoch, save_epochs, coords="polarrel", name=None, losses_path=None, show=False
):
    """Evaluation metric plots per epoch"""
    if coords == "cartesian":
        plabels = ["$p_x$ (GeV)", "$p_y$ (GeV)", "$p_z$ (GeV)"]
    elif coords == "polarrel":
        plabels = ["$\eta^{rel}$", "$\phi^{rel}$", "$p_T^{rel}$"]
    elif coords == "polarrelabspt":
        plabels = ["$\eta^{rel}$", "$\phi^{rel}$", "$p_T (GeV)$"]
    # jlabels = ['Relative Mass', 'Relative $p_T$', 'EFP']
    colors = ["blue", "green", "orange", "red", "yellow"]

    x = np.arange(0, epoch + 1, save_epochs)[-len(losses["w1p"]) :]

    fig = plt.figure(figsize=(30, 24))

    if "w1p" in losses:
        for i in range(3):
            fig.add_subplot(3, 3, i + 1)
            plt.plot(x, np.array(losses["w1p"])[:, i])
            plt.xlabel("Epoch")
            plt.ylabel("Particle " + plabels[i] + " $W_1$")
            plt.yscale("log")

    # x = np.arange(0, epoch + 1, args.save_epochs)[-len(losses['w1j_' + str(args.w1_num_samples[0]) + 'm']):]

    if "w1m" in losses:
        fig.add_subplot(3, 3, 4)
        plt.plot(x, np.array(losses["w1m"])[:, 0])
        plt.xlabel("Epoch")
        plt.ylabel("Jet Relative Mass $W_1$")
        plt.yscale("log")

    if "w1efp" in losses:
        fig.add_subplot(3, 3, 5)
        for i in range(5):
            plt.plot(x, np.array(losses["w1p"])[:, i], label="EFP " + str(i + 1), color=colors[i])
        plt.legend(loc=1)
        plt.xlabel("Epoch")
        plt.ylabel("Jet EFPs $W_1$")
        plt.yscale("log")

    if "mmd" in losses and "coverage" in losses:
        # x = x[-len(losses['mmd']):]
        metrics = {"mmd": (1, "MMD"), "coverage": (2, "Coverage")}
        for key, (i, label) in metrics.items():
            fig.add_subplot(3, 3, 6 + i)
            plt.plot(x, np.array(losses[key]))
            plt.xlabel("Epoch")
            plt.ylabel(label)
            if key == "mmd":
                plt.yscale("log")

    if "fpnd" in losses:
        fig.add_subplot(3, 3, 9)
        plt.plot(x, np.array(losses["fpnd"]))
        plt.xlabel("Epoch")
        plt.ylabel("FPND")
        plt.yscale("log")

    if losses_path is not None and name is not None:
        plt.savefig(losses_path + name + ".pdf", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
