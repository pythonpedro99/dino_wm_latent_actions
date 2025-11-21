FROM nvidia/cuda:12.2.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget bzip2 ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc && \
    echo "conda activate dino_wm" >> /etc/bash.bashrc

WORKDIR /workspace

COPY environment.yaml /tmp/environment.yaml
RUN conda update -n base -c defaults conda && \
    conda env create -f /tmp/environment.yaml && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=dino_wm
ENV CONDA_PREFIX=/opt/conda/envs/dino_wm
ENV PATH=${CONDA_PREFIX}/bin:/opt/conda/bin:$PATH

COPY . /workspace

CMD ["bash"]
