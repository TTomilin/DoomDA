import gym


class DoomDefendRewardShaping(gym.Wrapper):
    """Reward shaping specific for gathering scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self._prev_ammo = None
        self.orig_env_reward = 0.0

    def _reward_shaping(self, info, done):
        if done:
            print('DONE')
        if info is None or done:
            return 0.0

        curr_health = info.get('HEALTH', 0.0)
        curr_ammo = info.get('AMMO2', 0.0)
        curr_kills = info.get('KILLCOUNT', 0.0)
        reward = 0.0

        if self._prev_health is not None:
            delta = self._prev_health - curr_health
            if delta < 0.0:
                reward = -0.1

        if self._prev_ammo is not None:
            delta = self._prev_ammo - curr_ammo
            if delta > 0.0:
                reward = -0.1

        if self._prev_kills is not None:
            delta = self._prev_kills - curr_kills
            if delta > 0.0:
                # reward = + 1.0
                # Reward currently configured in-game
                pass

        self._prev_ammo = curr_ammo
        self._prev_kills = curr_kills
        return reward

    def reset(self):
        self._prev_ammo = None
        self.orig_env_reward = 0.0
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.orig_env_reward += reward
        reward += self._reward_shaping(info, done)

        if done:
            true_reward = self.orig_env_reward
            info['true_reward'] = true_reward

        return observation, reward, done, info
