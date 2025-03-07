# Use NVIDIA's official CUDA 11.2.1 base image with Ubuntu 20.04
FROM ubuntu:latest

# Set the working directory
WORKDIR /app

# Install necessary dependencies
RUN apt-get update 
RUN apt-get install -y \
    bash \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libnccl2 \
    libnccl-dev \
    wget \ 
    nano \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install wget to fetch Miniconda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

    # Install Miniconda on x86 or ARM platforms
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version


# Set NVIDIA runtime for the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


# Ensure CUDA binaries and Python are in PATH
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Currently, we are mounting the code directory to the container instead of copying it
# COPY . /app

# Use bash and activate the environment in CMD
CMD ["/bin/bash", "-c", "tail -f /dev/null"]

# TO BUILD, Use :
# docker build -t yue-postproc .
# docker run --gpus all --runtime=nvidia -it -v /path/to/code:/app --name YuE-PostProc yue-postproc
