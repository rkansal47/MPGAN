FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest

LABEL maintainer="Raghav Kansal <rkansal@ucsd.edu>"

RUN sudo apt-get update
RUN sudo apt-get update
RUN sudo apt-get install -y vim screen wget rsync gcc g++

RUN pip install setGPU
RUN pip install vector mplhep awkward coffea
RUN pip install energyflow wasserstein pot

RUN pip install guppy3

RUN pip install qpth cvxpy

RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu114.html

RUN pip install jetnet

RUN pip install numba

# Set the default command to python3.
CMD ["python3"]