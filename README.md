# gail-tf
Tensorflow implementation of Generative Adversarial Imitation Learning (and 
behavior cloning)

**disclaimers**: some code is borrowed from @openai/baselines

## What's GAIL?
- model free imtation learning -> low sample efficiency in training time
  - model-based GAIL: End-to-End Differentiable Adversarial Imitation Learning
- Directly extract policy from demonstrations
- Remove the RL optimization from the inner loop od inverse RL
- Some work based on GAIL:
  - Inferring The Latent Structure of Human Decision-Making from Raw Visual 
    Inputs
  - Multi-Modal Imitation Learning from Unstructured Demonstrations using 
  Generative Adversarial Nets
  - Robust Imitation of Diverse Behaviors
  
## Requirements
- python==3.5.2
- mujoco-py==0.5.7
- tensorflow==1.1.0
- gym==0.9.3

## Run the code
I separate the code into two parts: (1) Sampling expert data, (2) Imitation 
learning with GAIL/BC

### Step 1: Generate expert data

#### Train the expert policy using PPO/TRPO, from openai/baselines
Ensure that `$GAILTF` is set to the path to your gail-tf repository, and 
`$ENV_ID` is any valid OpenAI gym environment (e.g. Hopper-v1, HalfCheetah-v1, 
etc.)

##### Configuration
``` bash
export GAILTF=/path/to/your/gail-tf
export ENV_ID="Hopper-v1"
export BASELINES_PATH=$GAILTF/gailtf/baselines/ppo1 # use gailtf/baselines/trpo_mpi for TRPO
export SAMPLE_STOCHASTIC="False"            # use True for stochastic sampling
export STOCHASTIC_POLICY="False"            # use True for a stochastic policy
export PYTHONPATH=$GAILTF:$PYTHONPATH       # as mentioned below
cd $GAILTF
```

##### Train the expert
```bash
python3 $BASELINES_PATH/run_mujoco.py --env_id $ENV_ID
```

The trained model will save in ```./checkpoint```, and its precise name will
vary based on your optimization method and environment ID. Choose the last 
checkpoint in the series.

```bash
export PATH_TO_CKPT=./checkpoint/trpo.Hopper.0.00/trpo.Hopper.00-900
```

##### Sample from the generated expert policy
```bash
python3 $BASELINES_PATH/run_mujoco.py --env_id $ENV_ID --task sample_trajectory --sample_stochastic $SAMPLE_STOCHASTIC --load_model_path $PATH_TO_CKPT
```

This will generate a pickle file that store the expert trajectories in 
```./XXX.pkl``` (e.g. deterministic.ppo.Hopper.0.00.pkl)

```bash
export PICKLE_PATH=./stochastic.trpo.Hopper.0.00.pkl
```

### Step 2: Imitation learning

#### Imitation learning via GAIL

```bash
python3 main.py --env_id $ENV_ID --expert_path $PICKLE_PATH
```

Usage:
```bash
--env_id:          The environment id
--num_cpu:         Number of CPU available during sampling
--expert_path:     The path to the pickle file generated in the [previous section]()
--traj_limitation: Limitation of the exerpt trajectories
--g_step:          Number of policy optimization steps in each iteration
--d_step:          Number of discriminator optimization steps in each iteration
--num_timesteps:   Number of timesteps to train (limit the number of timesteps to interact with environment)
```

To view the summary plots in TensorBoard, issue
```bash
tensorboard --logdir $GAILTF/log
```

##### Evaluate your GAIL agent
```bash
python3 main.py --env_id $ENV_ID --task evaluate --stochastic_policy $STOCHASTIC_POLICY --load_model_path $PATH_TO_CKPT --expert_path $PICKLE_PATH
```

#### Imitation learning via Behavioral Cloning
```bash
python3 main.py --env_id $ENV_ID --algo bc --expert_path $PICKLE_PATH
```

##### Evaluate your BC agent
```bash
python3 main.py --env_id $ENV_ID --algo bc --task evalaute --stochastic_policy $STOCHASTIC_POLICY --load_model_path $PATH_TO_CKPT --expert_path $PICKLE_PATH
```

## Results

Note: The following hyper-parameter setting is the best that I've tested (simple 
grid search on setting with 1500 trajectories), just for your information.

The different curves below correspond to different expert size (1000,100,10,5).

- Hopper-v1 (Average total return of expert policy: 3589)

```bash
python3 main.py --env_id Hopper-v1 --expert_path baselines/ppo1/deterministic.ppo.Hopper.0.00.pkl --g_step 3 --adversary_entcoeff 0
```

![](misc/Hopper-true-reward.png)

- Walker-v1 (Average total return of expert policy: 4392)

```bash
python3 main.py --env_id Walker2d-v1 --expert_path baselines/ppo1/deterministic.ppo.Walker2d.0.00.pkl --g_step 3 --adversary_entcoeff 1e-3
```

![](misc/Walker2d-true-reward.png)

- HalfCheetah-v1 (Average total return of expert policy: 2110)

For HalfCheetah-v1 and Ant-v1, using behavior cloning is needed:
```bash
python3 main.py --env_id HalfCheetah-v1 --expert_path baselines/ppo1/deterministic.ppo.HalfCheetah.0.00.pkl --pretrained True --BC_max_iter 10000 --g_step 3 --adversary_entcoeff 1e-3
```

![](misc/HalfCheetah-true-reward.png)

**You can find more details [here](https://github.com/andrewliao11/gail-tf/blob/master/misc/exp.md), 
GAIL policy [here](https://drive.google.com/drive/folders/0B3fKFm-j0RqeRnZMTUJHSmdIdlU?usp=sharing), 
and BC policy [here](https://drive.google.com/drive/folders/0B3fKFm-j0RqeVFFmMWpHMk85cUk?usp=sharing)**

## Hacking
We don't have a pip package yet, so you'll need to add this repo to your 
PYTHONPATH manually.
```bash
export PYTHONPATH=/path/to/your/repo/with/gailtf:$PYTHONPATH
```

## TODO
* Create pip package/setup.py
* Make style PEP8 compliant
* Create requirements.txt
* Depend on openai/baselines directly and modularize modifications
* openai/robotschool support


## Reference
- Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning, [[arxiv](https://arxiv.org/abs/1606.03476)]
- @openai/imitation
- @openai/baselines
