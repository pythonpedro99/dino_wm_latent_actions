# Action-Label-Free World-Model Planning: Extending
DINO-WM with Inverse Dynamics
[[Thesis]](https://arxiv.org/abs/2411.04983) [[Code]]() [[Datasets]](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) 

[Julian Quast](https://gaoyuezhou.github.io/), Technische Universität Berlin

![teaser_figure](assets/trainings_architecture.png)

# Contents 

1. [Installation](#installation)
2. [Datasets](#datasets)
3. [Train a DINO-WM](#train-a-dino-wm)
4. [Plan with a DINO-WM](#plan-with-a-dino-wm)

## Installation

TBD

# Datasets

Dataset for each task can be downloaded [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). 

Once the datasets are downloaded, unzip them.

Set an environment variable pointing to your dataset folder:
```bash
# Replace /path/to/data with the actual path to your dataset folder.
export DATASET_DIR=/path/to/data
```
Inside the dataset folder, you should find the following structure:
```
data
├── deformable
│   ├── granular
│   └── rope
├── point_maze
├── pusht_noise
└── wall_single
```


# Train a inverse dynamics DINO-WM
Once you have completed the above steps, you can check whether you could launch training with an example command like this:

```
python train.py --config-name train.yaml env=point_maze frameskip=5 num_hist=3
```
You may specify models' output directory at `ckpt_base_path` in `conf/train.yaml`.

# Plan with a DINO-WM
Once a world model has been trained, you may use it for planning with an example command like this:

```
python plan.py model_name=<model_name> n_evals=5 planner=cem goal_H=5 goal_source='random_state' planner.opt_steps=30
```

where the model is saved at folder `<ckpt_base_path>/outputs/<model_name>`, and `<ckpt_base_path>` can be specified in `conf/plan.yaml`.


# Pre-trained Model Checkpoints

We have uploaded our trained world model checkpoints for  PushT [here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28) under `checkpoints`. You can launch planning jobs with their respective configs in the repo:

First, update `ckpt_base_path` to where the checkpoints are saved in the plan configs.

Then launch planning runs with the following commands:
```bash
# PointMaze
python plan.py --config-name plan_point_maze.yaml model_name=point_maze
# PushT
python plan.py --config-name plan_pusht.yaml model_name=pusht
# Wall
python plan.py --config-name plan_wall.yaml model_name=wall
```

Planning logs and visualizations can be found in `./plan_outputs`.


## Citation

```
@misc{zhou2024dinowmworldmodelspretrained,
      title={DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning}, 
      author={Gaoyue Zhou and Hengkai Pan and Yann LeCun and Lerrel Pinto},
      year={2024},
      eprint={2411.04983},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.04983}, 
}
```
