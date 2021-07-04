Download the [JetNet](https://zenodo.org/record/4834876#.YOIyni1w1hE) dataset, convert from `csv` to PyTorch format - reshaped as `[N, 30, 4]` shape tensors, and save them here. e.g. for gluon jets:

`torch.save(torch.tensor(np.loadtxt('g_jets.csv').reshape(-1, 30, 4)), 'datasets/g_jets.pt')`
