FROM anibali/pytorch:1.5.0-cuda10.2

RUN sudo apt-get update
RUN sudo apt-get install -y vim
RUN sudo apt-get install -y screen
RUN sudo apt-get install -y gcc
RUN sudo apt-get install -y wget

RUN pip install setGPU
RUN pip install scikit-image

# Install PyTorch Geometric. UPDATE: can be now installed more easily using conda https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
RUN CPATH=/usr/local/cuda/include:$CPATH \
 && LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
 && DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH

ENV TORCH_VERSION="1.5.0"
ENV CUDA_VERSION="cu102"

RUN pip install torch-scatter==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html \
 && pip install torch-sparse==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html \
 && pip install torch-cluster==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html \
 && pip install torch-spline-conv==latest+${CUDA_VERSION} -f https://pytorch-geometric.com/whl/torch-${TORCH_VERSION}.html \
 && pip install torch-geometric

# More libraries
RUN pip install mplhep
RUN pip install scikit-hep
RUN pip install scikit-learn
RUN pip install energyflow
RUN pip install coffea
RUN pip install guppy3
RUN pip install awkward
RUN pip install wasserstein
RUN pip install pot

# Set the default command to python3.
CMD ["python3"]
