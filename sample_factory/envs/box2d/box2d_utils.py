import gym

from sample_factory.envs.env_wrappers import RecordingWrapper


class Box2DSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


BOX2D_ENVS = [
    Box2DSpec('box2d_lunarlander', 'LunarLander-v2'),
    Box2DSpec('box2d_lunarlandercont', 'LunarLanderContinuous-v2'),
]


def box2d_env_by_name(name):
    for cfg in BOX2D_ENVS:
        if cfg.name == name:
            return cfg
    raise ValueError('Unknown Box2D env')


class Box2DRecordingWrapper(RecordingWrapper):

    def __init__(self, env, record_to):
        super().__init__(env, record_to)

    # noinspection PyMethodOverriding
    def render(self, mode, **kwargs):
        self.env.render()
        frame = self.env.render('rgb_array')
        self._record(frame)
        return frame

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._recorded_episode_reward += reward
        return observation, reward, done, info


def make_box2d_env(env_name, cfg=None, **kwargs):
    box2d_spec = box2d_env_by_name(env_name)
    env = gym.make(box2d_spec.env_id)
    return env
