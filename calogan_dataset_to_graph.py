import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak

myfile = uproot.open("./datasets/testCalo.root")
mytree = myfile["cell_tree"]
mydata = mytree.arrays()


LAYER_SPECS = [(3, 96), (12, 12), (12, 6)]
Lnt = [LAYER_SPECS[i][0] * LAYER_SPECS[i][1] for i in range(3)]

# plt.hist(ak.count(mydata["EnergyVector"], axis=1), bins=np.linspace(0, 100, 101))


def ak_to_np(arr: ak.Array, clip_target: int = 100):
    return ak.fill_none(ak.pad_none(arr, clip_target, axis=1, clip=True), 0, axis=None).to_numpy()


layers_pos = []
layers_energy = []

# The first is the total energy
layers_pos.append(
    ak_to_np(
        mydata["PositionVector"][
            (mydata["PositionVector"] < Lnt[0]) * (mydata["PositionVector"] >= 0)
        ]
    ).astype(int)
)


layers_energy.append(
    ak_to_np(
        mydata["EnergyVector"][
            (mydata["PositionVector"] < Lnt[0]) * (mydata["PositionVector"] >= 0)
        ]
    )
)

layers_pos.append(
    ak_to_np(
        mydata["PositionVector"][
            (mydata["PositionVector"] < Lnt[0] + Lnt[1]) * (mydata["PositionVector"] >= Lnt[0])
        ]
        - Lnt[0]
    ).astype(int)
)

layers_energy.append(
    ak_to_np(
        mydata["EnergyVector"][
            (mydata["PositionVector"] < Lnt[0] + Lnt[1]) * (mydata["PositionVector"] >= Lnt[0])
        ]
    )
)

# The last three are overflow
layers_pos.append(
    ak_to_np(
        mydata["PositionVector"][
            (mydata["PositionVector"] >= Lnt[0] + Lnt[1])
            * (mydata["PositionVector"] < Lnt[0] + Lnt[1] + Lnt[2])
        ]
        - Lnt[0]
        - Lnt[1]
    ).astype(int)
)

layers_energy.append(
    ak_to_np(
        mydata["EnergyVector"][
            (mydata["PositionVector"] >= Lnt[0] + Lnt[1])
            * (mydata["PositionVector"] < Lnt[0] + Lnt[1] + Lnt[2])
        ]
    )
)


layers_pos2d = []


# convert 1D coordinates into 2D, and pad z value
for i in range(len(LAYER_SPECS)):
    phi, eta = np.unravel_index(layers_pos[i], LAYER_SPECS[i])
    phi = (phi + 0.5) / (LAYER_SPECS[i][0])
    eta = (eta + 0.5) / (LAYER_SPECS[i][1])

    layers_pos2d.append(
        np.pad(
            np.stack((eta, phi), axis=-1),
            ((0, 0), (0, 0), (0, 1)),
            constant_values=(i + 0.5) / 3.0,
        )
    )


layers = []

for i in range(len(LAYER_SPECS)):
    layers.append(np.concatenate((layers_pos2d[i], layers_energy[i][..., np.newaxis]), axis=2))

dataset = np.concatenate(layers, axis=1)

NUM_HITS = 30

clipped_dataset = np.array(list(map(lambda x: x[x[:, 3].argsort()][-NUM_HITS:], dataset)))[:, ::-1]


mask = (clipped_dataset[:, :, 3:] != 0).astype(float)

final_dataset = np.concatenate((clipped_dataset, mask), axis=2)

np.save("./datasets/calogan_graph_30_hits_etaphizEmask.npy", final_dataset)


final_dataset

# plt.hist(final_dataset[:)

final_dataset.shape
mask.shape


hits = final_dataset[mask[:, :, 0].astype(bool)]

plt.hist(hits[:, 3])
plt.yscale("log")


fig = plt.figure(figsize=(22, 8))

for i in range(3):
    fig.add_subplot(1, 3, i + 1)
    plt.ticklabel_format(axis="y", scilimits=(0, 0), useMathText=True)
    _ = plt.hist(parts_real[:, i], pbins[i], histtype="step", label="Real", color="red")
    _ = plt.hist(parts_gen[:, i], pbins[i], histtype="step", label="Generated", color="blue")
