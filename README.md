# DoomDA

DoomDA is a benchmark intended for research in generalization of deep reinforcement learning agents. The benchmark is
built upon [ViZDoom](https://github.com/mwydmuch/ViZDoom), which is addressed to pixel based learning in the FPS game
domain.

## Installation

DoomDA is supported by Linux and macOS only.

- To install VizDoom just follow system setup instructions from the original repository 
([ViZDoom](https://github.com/mwydmuch/ViZDoom)), after which the latest VizDoom can be 
installed from PyPI: ```pip install vizdoom```. Version 1.1.9 or above is recommended.

- Clone the DoomDA repo: `git clone https://github.com/TTomilin/DoomDA.git`

### Local

- Install from PyPI: `pip3 install -e ./DoomDA`

### Conda

- Create and activate conda env:

```
conda env create -f ./DoomDA/environment.yml
conda activate doom-da
```

#### Remote

Under some circumstances you may run into issues when attempting to import local modules since your job scheduling system
copies the submitted script to a location on the selected partition. This can be fixed by adding the following lines
after the hashbang and before any other imports in your script:

```
import os 
import sys 
sys.path.append(os.getcwd())
```

## Environments

The benchmark currently consists of 3 scenarios, each with environments of single and combined modifications.

### Defend the Center

#### Single Modification
  * Gore
  * Mossy Bricks
  * Stone Wall
  * Flying Enemies
  * Fuzzy Enemies
  * Resized Enemies
#### Double Modification
  * Gore + Mossy Bricks
  * Resized Enemies + Fuzzy Enemies
  * Stone Wall + Flying Enemies
  * Resized Enemies + Gore

### Health Gathering

#### Single Modification
  * Lava
  * Water
  * Slime
  * Poison
  * Obstacles
  * Stimpacks
  * Short Agent
  * Shaded Kits
  * Resized kits
#### Double Modification
  * Slime + Obstacles
  * Shaded Stimpacks
  * Resized Kits + Lava
  * Short Agent + Water
  * Resized Shaded Kits
  * Stimpacks + Obstacles

### Dodge Projectiles

#### Single Modification
  * Barons
  * Mancubus
  * Flames
  * Cacodemons
  * Resized Agent
  * Flaming Skulls
  * City
#### Double Modification
  * City + Resized Agent
  * Barons + Flaming Skulls
  * Cacodemons + Flames
  * Mancubus +  Resized Agent


## Using DoomDA

Once DoomDA is installed, it defines two main entry points, one for training, and one for algorithm evaluation:

* `sample_factory.algorithms.appo.train_appo`
* `sample_factory.algorithms.appo.enjoy_appo`

The environments listed above are already added to the env registry in the default installation, 
so training on these environments is as simple as providing basic configuration parameters:

```
python -m sample_factory.algorithms.appo.train_appo --env=doom_defend_the_center --algo=APPO --experiment=stone_wall_appo --task stone_wall --train_for_env_steps=3000000 --num_workers=20 --num_envs_per_worker=20
python -m sample_factory.algorithms.appo.enjoy_appo --env=doom_defend_the_center --algo=APPO --experiment=stone_wall_appo --task stone_wall
```

### Available Arguments

Here we provide command lines which serve as an example on how to configure large-scale RL experiments of various
environments.

| Argument                             |        Default Value        | Description                                                                                                                                                                                                                                                                                                                                                                                                    |
|--------------------------------------|:---------------------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| --algo                               |            APPO             | Algo type to use (pass "APPO" if in doubt)')                                                                                                                                                                                                                                                                                                                                                                   |
| --env                                |            None             | Fully-qualified environment name in the form envfamily_envname, e.g. doom_defend_the_center')                                                                                                                                                                                                                                                                                                                  |
| --experiment                         |           default           | Unique experiment name. This will also be the name for the experiment folder in the train dir. If the experiment folder with this name already exists the experiment will be RESUMED! Any parameters passed from command line that do not match the parameters stored in the experiment cfg.json file will be overridden.                                                                                      | 
| --experiments_root                   |            None             | If not None, store experiment data in the specified subfolder of train_dir. Useful for groups of experiments (e.g. gridsearch)                                                                                                                                                                                                                                                                                 |
| -h --help                            |              -              | Print the help message                                                                                                                                                                                                                                                                                                                                                                                         |
| --experiment_summaries_interval      |             20              | How often in seconds we write avg. statistics about the experiment (reward, episode length, extra stats...)                                                                                                                                                                                                                                                                                                    |
| --adam_eps                           |            1e-6             | Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)                                                                                                                                                                                                                                                                                                                    |
| --adam_beta1                         |             0.9             | Adam momentum decay coefficient                                                                                                                                                                                                                                                                                                                                                                                |
| --adam_beta2                         |            0.999            | Adam second momentum decay coefficient                                                                                                                                                                                                                                                                                                                                                                         |
| --gae_lambda                         |            0.95             | Generalized Advantage Estimation discounting (only used when V-trace is False                                                                                                                                                                                                                                                                                                                                  |
| --rollout                            |             32              | Length of the rollout from each environment in timesteps. Once we collect this many timesteps on actor worker, we send this trajectory to the learner. The length of the rollout will determine how many timesteps are used to calculate bootstrapped'                                                                                                                                                         |
| --num_workers                        | multiprocessing.cpu_count() | Number of parallel environment workers. Should be less than num_envs and should divide num_envs                                                                                                                                                                                                                                                                                                                |
| --recurrence                         |             32              | Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly. For V-trace recurrence should be equal to rollout length.                                                                                                                                                                                        |
| --use_rnn                            |            True             | Whether to use RNN core in a policy or not                                                                                                                                                                                                                                                                                                                                                                     |
| --rnn_type                           |            'gru'            | Type of RNN cell to use if use_rnn is True                                                                                                                                                                                                                                                                                                                                                                     |
| --rnn_num_layers                     |              1              | Number of RNN layers to use if use_rnn is True                                                                                                                                                                                                                                                                                                                                                                 |
| --ppo_clip_ratio                     |            0.1,             | We use unbiased clip(x, 1+e, 1/(1+e)) instead of clip(x, 1+e, 1-e) in the paper                                                                                                                                                                                                                                                                                                                                |
| --ppo_clip_value                     |            1.0,             | Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude                                                                                                                                                                                                                                                                                                                    |
| --batch_size                         |            1024             | Minibatch size for SGD                                                                                                                                                                                                                                                                                                                                                                                         |
| --num_batches_per_iteration          |              1              | How many minibatches we collect before training on the collected experience. It is generally recommended to set this to 1 for most experiments, because any higher value will increase the policy lag                                                                                                                                                                                                          |
| --ppo_epochs                         |              1              | Number of training epochs before a new batch of experience is collected                                                                                                                                                                                                                                                                                                                                        |
| --num_minibatches_to_accumulate      |             -1              | This parameter governs the maximum number of minibatches the learner can accumulate before further experience collection is stopped.                                                                                                                                                                                                                                                                           |
| --max_grad_norm                      |             4.0             | Max L2 norm of the gradient vector                                                                                                                                                                                                                                                                                                                                                                             |
| --exploration_loss_coeff             |            0.003            | Coefficient for the exploration component of the loss function.                                                                                                                                                                                                                                                                                                                                                |
| --value_loss_coeff                   |             0.5             | Coefficient for the critic loss                                                                                                                                                                                                                                                                                                                                                                                |
| --kl_loss_coeff                      |            0.0,             | Coefficient for fixed KL loss (as used by Schulman et al. in https://arxiv.org/pdf/1707.06347.pdf). Highly recommended for environments with continuous action spaces.                                                                                                                                                                                                                                         |
| --exploration_loss                   |           entropy           | Usually the exploration loss is based on maximizing the entropy of the probability                                                                                                                                                                                                                                                                                                                             |
| --num_envs_per_worker                |              2              | Number of envs on a single CPU actor, in high-throughput configurations this should be in 10-30 range for ViZDoom. Must be even for double-buffered sampling                                                                                                                                                                                                                                                   |
| --worker_num_splits                  |              2              | Typically we split a vector of envs into two parts for "double buffered" experience collection. Set this to 1 to disable double buffering. Set this to 3 for triple buffering!                                                                                                                                                                                                                                 |
| --num_policies                       |              1              | Number of policies to train jointly                                                                                                                                                                                                                                                                                                                                                                            |
| --policy_workers_per_policy          |              1              | Number of policy workers that compute forward pass (per policy)                                                                                                                                                                                                                                                                                                                                                |
| --max_policy_lag                     |            10000            | Max policy lag in policy versions. Discard all experience that is older than this. This should be increased for configurations with multiple epochs of SGD because naturally policy-lag may exceed this value.                                                                                                                                                                                                 |
| --traj_buffers_excess_ratio          |             1.3             | Increase this value to make sure the system always has enough free trajectory buffers (can be useful when i.e. a lot of inactive agents in multi-agent envs). Decrease this to 1.0 to save as much RAM as possible.                                                                                                                                                                                            |
| --decorrelate_experience_max_seconds |             10              | Decorrelating experience serves two benefits. First: this is better for learning because samples from workers come from random moments in the episode, becoming more "i.i.d".                                                                                                                                                                                                                                  |
| --decorrelate_envs_on_one_worker     |            True             | In addition to temporal decorrelation of worker processes, also decorrelate envs within one worker process. For environments with a fixed episode length it can prevent the reset from happening in the same rollout for all envs simultaneously, which makes experience collection more uniform.                                                                                                              |
| --with_vtrace                        |            True             | Enables V-trace off-policy correction. If this is True, then GAE is not used                                                                                                                                                                                                                                                                                                                                   |
| --vtrace_rho                         |             1.0             | rho_hat clipping parameter of the V-trace algorithm (importance sampling truncation)                                                                                                                                                                                                                                                                                                                           |
| --vtrace_c                           |             1.0             | c_hat clipping parameter of the V-trace algorithm. Low values for c_hat can reduce variance of the advantage estimates (similar to GAE lambda < 1)                                                                                                                                                                                                                                                             |
| --set_workers_cpu_affinity           |            True             | Whether to assign workers to specific CPU cores or not. The logic is beneficial for most workloads because prevents a lot of context switching. However for some environments it can be better to disable it, to allow one worker to use all cores some of the time. This can be the case for some DMLab environments with very expensive episode reset that can use parallel CPU cores for level generation.  |
| --force_envs_single_thread           |            True             | Some environments may themselves use parallel libraries such as OpenMP or MKL. Since we parallelize environments on the level of workers, there is no need to keep this parallel semantic. This flag uses threadpoolctl to force libraries such as OpenMP and MKL to use only a single thread within the environment. Default value (True) is recommended unless you are running fewer workers than CPU cores. |
| --reset_timeout_seconds              |             120             | Fail worker on initialization if not a single environment was reset in this time (worker probably got stuck)                                                                                                                                                                                                                                                                                                   |
| --default_niceness                   |              0              | Niceness of the highest priority process (the learner). Values below zero require elevated privileges.                                                                                                                                                                                                                                                                                                         |
| --train_in_background_thread         |            True             | Using background thread for training is faster and allows preparing the next batch while training is in progress. Unfortunately debugging can become very tricky in this case. So there is an option to use only a single thread on the learner to simplify the debugging.                                                                                                                                     |
| --learner_main_loop_num_cores        |              1              | When batching on the learner is the bottleneck, increasing the number of cores PyTorch uses can improve the performance                                                                                                                                                                                                                                                                                        |
| --actor_worker_gpus                  |             []              | By default, actor workers only use CPUs. Changes this if e.g. you need GPU-based rendering on the actors                                                                                                                                                                                                                                                                                                       |
| --with_pbt                           |            False            | Enables population-based training basic features                                                                                                                                                                                                                                                                                                                                                               |
| --pbt_mix_policies_in_one_env        |            True             | For multi-agent envs, whether we mix different policies in one env.                                                                                                                                                                                                                                                                                                                                            |
| --pbt_period_env_steps               |          int(5e6)           | Periodically replace the worst policies with the best ones and perturb the hyperparameters                                                                                                                                                                                                                                                                                                                     |
| --pbt_start_mutation                 |          int(2e7)           | Allow initial diversification, start PBT after this many env steps                                                                                                                                                                                                                                                                                                                                             |
| --pbt_replace_fraction               |             0.3             | A portion of policies performing worst to be replace by better policies (rounded up)                                                                                                                                                                                                                                                                                                                           |
| --pbt_mutation_rate                  |            0.15             | Probability that a parameter mutates                                                                                                                                                                                                                                                                                                                                                                           |
| --pbt_replace_reward_gap             |             0.1             | Relative gap in true reward when replacing weights of the policy with a better performing one                                                                                                                                                                                                                                                                                                                  |
| --pbt_replace_reward_gap_absolute    |            1e-6             | Absolute gap in true reward when replacing weights of the policy with a better performing one                                                                                                                                                                                                                                                                                                                  |
| --pbt_optimize_batch_size            |            False            | Whether to optimize batch size or not (experimental)                                                                                                                                                                                                                                                                                                                                                           |
| --pbt_optimize_gamma                 |            False            | Whether to optimize gamma, discount factor, or not (experimental)                                                                                                                                                                                                                                                                                                                                              |
| --pbt_target_objective               |         true_reward         | Policy stat to optimize with PBT. true_reward (default) is equal to raw env reward if not specified, but can also be any other per-policy stat. For DMlab-30 use value "dmlab_target_objective" (which is capped human normalized score)                                                                                                                                                                       |
| --pbt_perturb_min                    |            1.05             | When PBT mutates a float hyperparam, it samples the change magnitude randomly from the uniform distribution [pbt_perturb_min, pbt_perturb_max]                                                                                                                                                                                                                                                                 |
| --pbt_perturb_max                    |             1.5             | When PBT mutates a float hyperparam, it samples the change magnitude randomly from the uniform distribution [pbt_perturb_min, pbt_perturb_max]                                                                                                                                                                                                                                                                 |
| --use_cpc                            |            False            | Use CPC                                                                                                                                                                                                                                                                                                                                                                                                        |A as an auxiliary loss durning learning |
| --cpc_forward_steps                  |              8              | Number of forward prediction steps for CPC                                                                                                                                                                                                                                                                                                                                                                     |
| --cpc_time_subsample                 |              6              | Number of timesteps to sample from each batch. This should be less than recurrence to decorrelate experience.                                                                                                                                                                                                                                                                                                  |
| --cpc_forward_subsample              |              2              | Number of forward steps to sample for loss computation. This should be less than cpc_forward_steps to decorrelate gradients.                                                                                                                                                                                                                                                                                   |
| --sampler_only                       |            False            | Do not send experience to the learner, measuring sampling throughput                                                                                                                                                                                                                                                                                                                                           |
| --eval_episodes                      |              3              | Number of episodes to evaluate                                                                                                                                                                                                                                                                                                                                                                                 |
| --eval_frequency                     |          1_000_000          | Number of time steps after to perform evaluation                                                                                                                                                                                                                                                                                                                                                               |
| --eval_render                        |            False            | Enables the rendering of evaluation episodes                                                                                                                                                                                                                                                                                                                                                                   |


### Monitoring training sessions

DoomDA uses Tensorboard summaries. Run Tensorboard to monitor your 
experiment: `tensorboard --logdir=train_dir --port=6006`

Additionally, we provide a helper script that has nice command line interface to monitor the experiment folders using
wildcard masks: `python -m sample_factory.tb '*custom_experiment*' '*another*custom*experiment_name'`

#### WandB support

DoomDA also supports experiment monitoring with Weights and Biases. In order to setup WandB locally
run `wandb login` in the terminal ([WandB Quickstart](https://docs.wandb.ai/quickstart#1.-set-up-wandb)).

Example command line to run an experiment with WandB monitoring:

```
python -m sample_factory.algorithms.appo.train_appo --env=doom_health_gathering --algo=APPO --experiment=test_wandb --with_wandb=True --wandb_user=<your_wandb_user> --wandb_key=<your_wandb_api_key> --wandb_tags test benchmark doom appo
```

A total list of WandB settings:

```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_key: WandB API key. Might need to be specified if running from a remote server. (default: None)
--wandb_project: WandB "Project" (default: DoomDA)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

Once the experiment is started the link to the monitored session is going to be available in the logs (or by searching
in Wandb Web console).

### Plotting results

We have provided a script for plotting experiment results at `sample_factory/utils/plot_results.py`.

A total list of command line arguments:

```
--env_family: Name of the environment family e.g., 'doom', 'box2d'. Must be specified (default: None)
--env: Name of the environment. Must be specified (default: None)
--experiments: List of experiment names. Must be specified (default: [])
--metrics: List of the metrics to plot (default: 'reward')
--title: Title of the plot (default: None)
--n_ticks: Limit the number of values to plot (default: sys.maxsize)
--style: Matplotlib built-in style to be used for plotting (default: 'seaborn')
```

Plotting can be executed via the following steps:

1. Download the log of the statistic from Tensorboard in JSON format.
2. Store the log file with the following directory
   structure: `sample_facory/statistics/<env_family>/<env>/<metric>_<experiment>.json`
3. Run `python -m sample_factory.utils.plot_results.py --env_family <env_family> --env <env> --metrics <metric>`
4. The resulting figures will be stored in `sample_facory/plots/<env_family>/<env>/<metric>.png`.

#### Example command line usage

Plotting the reward and kill count of three ViZDoom experiments

```
python -m sample_factory.utils.plot_results.py --env_family doom --env defend_the_center --metrics reward kill_count --title DefendTheCenter --experiments default gore stone_wall
```