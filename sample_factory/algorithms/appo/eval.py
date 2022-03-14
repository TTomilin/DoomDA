import time
from typing import List

import numpy as np
import torch
from tensorboardX import SummaryWriter

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.multi_agent_wrapper import is_multiagent_env, MultiAgentWrapper
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict


def evaluate(cfg: AttrDict, writer: SummaryWriter, steps: int, tasks: List[str]):
    render_action_repeat = cfg.env_frameskip
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1

    last_render_start = time.time()

    policy_id = 0

    rnn = '_' + cfg.rnn_type if cfg.use_rnn else ''

    for task in tasks:

        episode_rewards = []
        frames_survived = []
        kill_counts = []
        ammo_left = []

        env = create_env(cfg.env, cfg=cfg, env_config=AttrDict({'worker_index': 0, 'vector_index': 0}))

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

        device = torch.device('cpu')  # Use CPU for evaluation to not consume training resources
        actor_critic.model_to_device(device)

        checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])

        is_multiagent = is_multiagent_env(env)
        if not is_multiagent:
            env = MultiAgentWrapper(env)

        if hasattr(env.unwrapped, 'reset_on_init'):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False

        scenario_path = env.unwrapped.scenario_path
        scenario_path = scenario_path.replace('default', task)
        env.unwrapped.scenario_path = scenario_path

        obs = env.reset()
        rnn_states = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)

        with torch.no_grad():
            for episode in range(cfg.eval_episodes):
                episode_reward = 0
                num_frames = 0
                done = False
                while not done:
                    obs_torch = AttrDict(transform_dict_observations(obs))
                    for key, x in obs_torch.items():
                        obs_torch[key] = torch.from_numpy(x).to(device).float()

                    policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

                    # sample actions from the distribution by default
                    actions = policy_outputs.actions

                    actions = actions.cpu().numpy()

                    rnn_states = policy_outputs.rnn_states

                    for _ in range(render_action_repeat):
                        if cfg.eval_render:
                            render(cfg, env, last_render_start)
                            last_render_start = time.time()

                        obs, rewards, dones, infos = env.step(actions)

                        # We only have one agent, so take the first element
                        info = infos[0]
                        done = dones[0]
                        episode_reward += rewards[0]

                        num_frames += 1

                        if done:
                            episode_rewards.append(episode_reward)
                            frames_survived.append(num_frames)
                            kill_counts.append(info.get('KILLCOUNT'))
                            ammo_left.append(info.get('AMMO2'))
                            log.info(f'Evaluation of task {task}. Episode {episode + 1} lasted {num_frames} frames. Reward: {episode_reward:.3f}. Steps: {steps}')
                            rnn_states = torch.zeros([1, get_hidden_size(cfg)], dtype=torch.float32, device=device)
                            episode_reward = 0
                            break

        writer.add_scalar(f'eval/reward{rnn}_{task}', np.mean(episode_rewards), steps)
        writer.add_scalar(f'eval/frames_survived{rnn}_{task}', np.mean(frames_survived), steps)
        if any(kill_counts):
            writer.add_scalar(f'eval/kill_count{rnn}_{task}', np.mean(kill_counts), steps)
        if any(ammo_left):
            writer.add_scalar(f'eval/ammo_left{rnn}_{task}', np.mean(ammo_left), steps)

        env.close()


def render(cfg: AttrDict, env, last_render_start: float):
    target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
    current_delay = time.time() - last_render_start
    time_wait = target_delay - current_delay
    if time_wait > 0:
        time.sleep(time_wait)
    env.render()
