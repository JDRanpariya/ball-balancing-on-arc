# Dockerfile for reproducing all paper results: figures, simulation evaluation,
# RL training, offline RL, world-model training, and Table I verification.
# Uses conda + env.yaml, which pins exact package versions (mirrored in
# requirements.lock.txt). The base image is pinned by digest below; the few
# apt build tools (gcc/g++/git/make) are intentionally left unpinned - they
# affect only the build toolchain, not the paper's numeric results.
# GPU not required; training works on CPU (slower for LSTM world-model training).
# MPPI uses GPU for inference if CUDA is available, falls back to CPU otherwise.
#
# Prerequisites - run BEFORE docker build:
#   git lfs pull   (materialise datasets, models, trial data)
#
# Build:
#   docker build -t balancer-repro .
#
# Common commands (see reproducibility_guide.md for full list):
#   Table I verify:        docker run --rm balancer-repro python paper/verify_table1.py
#   All figures:           docker run --rm -v "$(pwd)/output:/output" balancer-repro make figures-docker
#   Sim benchmark:         docker run --rm -v "$(pwd)/output:/output" balancer-repro make eval-sim-docker
#   MAE check:             docker run --rm balancer-repro python paper/check_mae.py
#   Interactive shell:     docker run --rm -it balancer-repro bash
#
# Note: passing a command (e.g. `make figures`) to `docker run` replaces the
# image's default CMD below, so the plain `figures`/`eval-sim` targets alone
# will NOT copy results to /output before the container exits. Use the
# `figures-docker` / `eval-sim-docker` Makefile targets instead - they run
# the real target and then copy results to /output when that mount exists.

FROM continuumio/miniconda3:24.11.1-0@sha256:6a66425f001f739d4778dd732e020afeb06175f49478fafc3ec673658d61550b

WORKDIR /app

# System deps for CasADi (gcc/g++) and matplotlib GL backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git git-lfs libgl1 libglib2.0-0 make && \
    rm -rf /var/lib/apt/lists/*

# Copy env.yaml first - conda layer is cached independently of code changes
COPY env.yaml .

# Create the conda environment from the pinned env.yaml (exact versions)
RUN conda env create -f env.yaml && conda clean -afy

# All subsequent RUN/CMD use the balancer conda env
SHELL ["conda", "run", "-n", "balancer", "/bin/bash", "-c"]

# Copy repo contents (run `git lfs pull` before `docker build` for real files)
COPY balancer/  balancer/
COPY paper/     paper/
COPY evaluation/ evaluation/
COPY training/  training/
COPY data/      data/
COPY Makefile   ./

# Install the balancer package into the conda env
RUN pip install --no-cache-dir -e "balancer/[all]"

# Activate balancer env in interactive shells
RUN echo "conda activate balancer" >> ~/.bashrc

# Default: regenerate all figures and copy to /output
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "balancer"]
CMD ["sh", "-c", "make figures && mkdir -p /output && \
     cp -r paper/ram/figures/* /output/ && \
     cp -r paper/ram/supplementary/figures /output/supplementary_figures"]
