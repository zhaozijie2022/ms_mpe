import gym
import mindspore.ops
from gym import spaces
import numpy as np
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space


def make_env(scenario_name):
    from mindspore_rl.environment.mpe.environment import MultiAgentEnv
    import mindspore_rl.environment.mpe.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


class MpeEnvironment(Environment):
    def __init__(self, params):
        # params = {'scenario': 'simple_spread'}
        super().__init__()
        self.params = params
        self._scenario = params['scenario']
        self._env = make_env(scenario_name=self._scenario)

        self._num_agent = self._env.n
        obs_dim = self._env.observation_space[0].shape[0]

        self._global_obs_dim = self._env.observation_space[0].shape[0] * self._num_agent

        self.step_info = {}

        self._observation_space = Space(
            (obs_dim,), np.float32, batch_shape=(self._num_agent,)
        )
        if isinstance(self._env.action_space[0], spaces.Discrete):
            action_dim = self._env.action_space[0].n
            self._action_space = Space(
                (1,), np.int32, low=0, high=action_dim, batch_shape=(self._num_agent,)
            )
        elif isinstance(self._env.action_space[0], spaces.Box):
            action_dim = self._env.action_space[0].shape[0]
            self._action_space = Space(
                (action_dim,), np.float32, batch_shape=(self._num_agent,)
            )
        else:
            raise ValueError("action space type can be only Discrete or Box")

        self._reward_space = Space((self._num_agent,), np.float32)
        self._done_space = Space((self._num_agent,), np.bool_, low=0, high=2)

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = [self._observation_space.ms_dtype]
        reset_output_shape = [self._observation_space.shape]
        self._reset_op = P.PyFunc(
            self._reset,
            reset_input_type, reset_input_shape,
            reset_output_type, reset_output_shape
        )

        # step op
        step_input_type = [self._action_space.ms_dtype]
        step_input_shape = [self._action_space.shape]
        step_output_type = [
            self._observation_space.ms_dtype,
            self._reward_space.ms_dtype,
            self._done_space.ms_dtype
        ]
        step_output_shape = [
            self._observation_space.shape,
            self._reward_space.shape,
            self._done_space.shape
        ]
        self._step_op = P.PyFunc(
            self._step,
            step_input_type, step_input_shape,
            step_output_type, step_output_shape
        )
        self.params['num_agent'] = self._num_agent
        self.params['episode_limit'] = 200

    def reset(self):
        return self._reset_op()

    def step(self, action):
        return self._step_op(action)

    def close(self):
        self._env.close()
        return True

    def _step(self, action):
        action_n = np.split(action, self.n)
        obs_n, reward_n, done_n, _ = self._env.step(action_n)
        obs = np.array(obs_n, self._observation_space.np_dtype)
        reward = np.array(reward_n, self._reward_space.np_dtype)
        done = np.array(done_n, self._done_space.np_dtype)
        return obs, reward, done

    def _reset(self):
        obs_n = self._env.reset()
        return np.array(obs_n)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def reward_space(self):
        return self._reward_space

    @property
    def done_space(self):
        return self._done_space

    @property
    def config(self):
        return self.params








