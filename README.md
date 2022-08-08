# MPGAN

[![DOI](https://zenodo.org/badge/382939833.svg)](https://zenodo.org/badge/latestdoi/382939833)
[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Code for Kansal et. al., *Particle Cloud Generation with Message Passing Generative Adversarial Networks*, NeurIPS 2021 [`arXiv:2106.11535`](https://arxiv.org/abs/2106.11535).


## Overview

This repository contains PyTorch code for the message passing GAN (MPGAN) [model](mpgan/model.py), as well as scripts for [training](train.py) the models from scratch, [generating](gen.py) and [plotting](plotting.py) the particle clouds. 
We include also [weights](trained_models) of fully trained models discussed in the paper. 

Additionally, we release the standalone [JetNet](https://github.com/jet-net/JetNet) library, which provides a PyTorch Dataset class for our JetNet dataset, implementations of the evaluation metrics discussed in the paper, and some more useful utilities for development in machine learning + jets.

## Dependencies

#### MPGAN Model

 - `torch >= 1.8.0`

#### Training, Plotting, Evaluation

 - `torch >= 1.8.0`
 - `0.1.2 >= jetnet >= 0.1.0` (*This branch hasn't been updated for later JetNet versions - see `main` branch for that*)
 - `numpy >= 1.21.0`
 - `matplotlib`
 - `mplhep`

#### External Models

 - `torch`
 - `torch_geometric`


A Docker image containing all necessary libraries can be found [here](https://gitlab-registry.nautilus.optiputer.net/raghsthebest/mnist-graph-gan:latest) ([Dockerfile](Dockerfile)).


## Training

Start training with:

```python
python train.py --name test_model --jets g [args]  
```

By default, model parameters, figures of particle and jet features, and plots of the training losses and evaluation metrics over time will be saved every five epochs in an automatically created `outputs/[name]` directory.

Some notes:
 - Will run on a GPU by default if available. 
 - The default arguments correspond to the final model architecture and training configuration used in the paper. 
 - Run `python train.py --help` or look at [setup_training.py](setup_training.py) for a full list of arguments.


## Generation

Pre-trained generators with saved state dictionaries and arguments can be used to generate samples with, for example:

```python
python gen.py --G-state-dict trained_models/mp_g/G_best_epoch.pt --G-args trained_models/mp_g/args.txt --num-samples 50,000 --output-file trained_models/mp_g/gen_jets.npy
```
