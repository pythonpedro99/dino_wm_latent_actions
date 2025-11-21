FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

SHELL ["/bin/bash", "-lc"]

# ======================================================================
# --- Base system dependencies ---
# Includes all libs needed for:
#   - OpenCV (libsm6, libxext6, libglib2.0-0, etc.)
#   - Shapely (libgeos-dev)
#   - Pygame/Pymunk rendering (X11 stack)
#   - Decord (libgl)
# ======================================================================
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
       wget \
       bzip2 \
       ca-certificates \
       git \
       # OpenGL / EGL / Video
       libgl1-mesa-glx \
       libosmesa6 \
       libglib2.0-0 \
       # X11 stack for pygame/opencv
       libxrender1 \
       libxext6 \
       libx11-6 \
       libxfixes3 \
       libxcb1 \
       libxi6 \
       libxtst6 \
       libxxf86vm1 \
       libsm6 \
       # Shapely GEOS fallback
       libgeos-dev \
       # Misc tooling
       patchelf \
       && rm -rf /var/lib/apt/lists/*

# ======================================================================
# --- Install Miniconda ---
# ======================================================================
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Activate conda in interactive shells
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate dino_wm" >> /etc/bash.bashrc

WORKDIR /workspace

# ======================================================================
# --- Copy minimal environment file ---
# ======================================================================
COPY environment.yaml /tmp/environment.yaml

# ======================================================================
# --- Remove Anaconda defaults (ToS clean), configure conda-forge ---
# ======================================================================
RUN conda config --system --remove channels defaults || true
RUN conda config --system --append channels conda-forge
RUN conda config --system --set channel_priority strict

# ======================================================================
# --- Create the dino_wm environment ---
# ======================================================================
RUN conda env create -f /tmp/environment.yaml && \
    conda clean -afy

# Export environment variables for runtime
ENV CONDA_DEFAULT_ENV=dino_wm
ENV CONDA_PREFIX=/opt/conda/envs/dino_wm
ENV PATH=${CONDA_PREFIX}/bin:/opt/conda/bin:$PATH

# ======================================================================
# --- Copy full repository into container ---
# ======================================================================
COPY . /workspace

CMD ["bash"]
