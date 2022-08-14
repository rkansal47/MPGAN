import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep


model = "20_t/"
loss_dir = f"outputs/{model}/losses/"
plots_dir = "plots/correlation_plots/"

loss_keys = ["fpnd", "mmd", "coverage"]

losses = {}

for key in loss_keys:
    losses[key] = np.loadtxt(loss_dir + key + ".txt")

losses["w1m"] = np.loadtxt(loss_dir + "w1m.txt")[:, 0]
losses["w1p"] = np.mean(np.loadtxt(loss_dir + "w1p.txt")[:, :3], axis=1)
losses["w1efp"] = np.mean(np.loadtxt(loss_dir + "w1efp.txt")[:, :5], axis=1)


def correlation_plot(xkey, ykey, xlabel, ylabel, range, scilimits=False):
    plt.rcParams.update({"font.size": 16})
    plt.style.use(hep.style.CMS)

    fig = plt.figure(figsize=(12, 10))
    h = plt.hist2d(losses[xkey], losses[ykey], bins=50, range=range, cmap="jet")
    if scilimits:
        plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    c = plt.colorbar(h[3])
    c.set_label("Number of batches")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{xlabel} vs {ylabel} Correlation")
    plt.savefig(f"{plots_dir}/{xkey}v{ykey}.pdf", bbox_inches="tight")
    plt.show()


correlation_plot("w1m", "fpnd", "W1-M", "FPND", [[0, 0.01], [0, 10]])
correlation_plot("w1m", "w1efp", "W1-M", "W1-EFP", [[0, 0.01], [0, 0.00025]], True)
correlation_plot("w1m", "w1p", "W1-M", "W1-P", [[0, 0.01], [0, 0.005]])
correlation_plot("w1p", "fpnd", "W1-P", "FPND", [[0, 0.005], [0, 10]])
correlation_plot("w1m", "mmd", "W1-M", "MMD", [[0, 0.01], [0, 0.1]])
correlation_plot("w1m", "coverage", "W1-M", "Coverage", [[0, 0.01], [0, 1]])


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses["w1m"], losses["fpnd"], bins=50, range=[[0, 0.02], [0, 50]], cmap="jet")
c = plt.colorbar(h[3])
c.set_label("Number of batches")
plt.xlabel("W1-M")
plt.ylabel("FPND")
plt.title("W1-M vs FPND Correlation")
plt.savefig(f"{plots_dir}/w1mvfpnd.pdf", bbox_inches="tight")


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses["w1m"], losses["w1efp"], bins=50, range=[[0, 0.015], [0, 0.0005]], cmap="jet")
plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label("Number of batches")
plt.xlabel("W1-M")
plt.ylabel("W1-EFP")
plt.title("W1-M vs W1-EFP Correlation")
plt.savefig(f"{plots_dir}/w1mvw1efp.pdf", bbox_inches="tight")


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses["w1m"], losses["w1p"], bins=50, range=[[0, 0.02], [0, 0.01]], cmap="jet")
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label("Number of batches")
plt.xlabel("W1-M")
plt.ylabel("W1-P")
plt.title("W1-M vs W1-P Correlation")
plt.savefig(f"{plots_dir}/w1mvw1p.pdf", bbox_inches="tight")


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses["w1p"], losses["fpnd"], bins=50, range=[[0, 0.01], [0, 50]], cmap="jet")
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label("Number of batches")
plt.xlabel("W1-P")
plt.ylabel("FPND")
plt.title("W1-P vs FPND Correlation")
plt.savefig(f"{plots_dir}/w1pvfpnd.pdf", bbox_inches="tight")


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses["w1m"], losses["mmd"], bins=50, range=[[0, 0.01], [0, 0.1]], cmap="jet")
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label("Number of batches")
plt.xlabel("W1-M")
plt.ylabel("MMD")
plt.title("W1-M vs MMD Correlation")
plt.savefig(f"{plots_dir}/w1mvmmd.pdf", bbox_inches="tight")


fig = plt.figure(figsize=(12, 10))
h = plt.hist2d(losses["w1m"], losses["coverage"], bins=50, range=[[0, 0.01], [0, 1]], cmap="jet")
# plt.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
c = plt.colorbar(h[3])
c.set_label("Number of batches")
plt.xlabel("W1-M")
plt.ylabel("COV")
plt.title("W1-M vs COV Correlation")
plt.savefig(f"{plots_dir}/w1mvcov.pdf", bbox_inches="tight")
