#!/bin/bash
#SBATCH --job-name=dino_wm_launch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu

module load apptainer

CONTAINER_IMAGE="${PWD}/dino_wm.sif"
BIND_PATH="/scratch/${USER}:/scratch/${USER},/home/${USER}:/home/${USER}"

apptainer exec --nv --bind "${BIND_PATH}" "${CONTAINER_IMAGE}" bash -lc "
    source /opt/conda/etc/profile.d/conda.sh
    conda activate dino_wm
    cd /workspace
    python train.py --multirun \
        hydra/launcher=submitit_slurm \
        hydra.launcher.nodes=5 \
        hydra.launcher.tasks_per_node=1 \
        hydra.launcher.gpus_per_node=1 \
        hydra.launcher.cpus_per_task=4 \
        hydra.launcher.mem_gb=16
"
