import gym
import mindspore.ops
from multiprocessing import Queue
from gym import spaces
import numpy as np
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space
from mindspore_rl.environment.env_process import EnvironmentProcess


def make_env(scenario_name, n_agents=3, n_landmarks=3):
    from .environment import MultiAgentEnv
    import mindspore_rl.environment.mpe.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario(n_agents=n_agents, n_landmarks=n_landmarks)
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


class MpeEnvironment(Environment):
    def __init__(self, params, env_id=0):
        # params = {"num": 10, "name": simple_spread, "proc_num": 32, "num_agent": 3}
        # env_num, scenario_name, process_num
        super().__init__()
        self.params = params
        self._nums = params['num']
        self._proc_num = params['proc_num']
        self._env_name = params['name']
        self._num_agent = params['num_agent']

        self._envs = []
        self.env_id = env_id

        for i in range(self._nums):
            mpe_env = make_env(scenario_name=self._env_name, n_agents=self._num_agent, n_landmarks=self._num_agent)
            mpe_env.seed(1 + i * 1000)
            self._envs.append(mpe_env)

        self._state_space = self._space_adapter(self._envs[0].observation_space[0], batch_shape=(self._num_agent,))
        self._action_space = self._space_adapter(self._envs[0].action_space[0], batch_shape=(self._num_agent,))
        self._reward_space = Space((1,), np.float32, batch_shape=(self._num_agent,))
        self._done_space = Space((1,), np.bool_, low=0, high=2, batch_shape=(self._num_agent,))

        step_input_shape = [(self._nums, self._num_agent, self._action_space.num_values)]
        step_output_shape = [((self._nums,) + self._state_space.shape),
                             ((self._nums,) + self._reward_space.shape),
                             ((self._nums,) + self._done_space.shape)]
        reset_output_shape = [((self._nums,) + self._state_space.shape)]

        self.step_ops = P.PyFunc(self._step,
                                 in_types=[self._action_space.ms_dtype, ],
                                 in_shapes=step_input_shape,
                                 out_types=[self._state_space.ms_dtype, self._reward_space.ms_dtype, self._done_space.ms_dtype],
                                 out_shapes=step_output_shape)

        self.reset_ops = P.PyFunc(self._reset, [], [],
                                  [self._state_space.ms_dtype, ],
                                  reset_output_shape)

        self.mpe_env_procs = []
        self.action_queues = []
        self.exp_queues = []
        self.init_state_queues = []

        if self._nums < self._proc_num:
            raise ValueError("Environment number can not be smaller than process number")

        avg_env_num_per_proc = int(self._nums / self._proc_num)
        for i in range(self._proc_num):
            action_q = Queue()
            self.action_queues.append(action_q)
            exp_q = Queue()
            self.exp_queues.append(exp_q)
            init_state_q = Queue()
            self.init_state_queues.append(init_state_q)

            assigned_env_num = i * avg_env_num_per_proc
            if assigned_env_num < self._nums:
                env_num = avg_env_num_per_proc
            else:
                env_num = self._nums - assigned_env_num

            env_proc = EnvironmentProcess(
                i, env_num, self._envs[env_num * i:env_num * (i + 1)], action_q, exp_q, init_state_q)
            self.mpe_env_procs.append(env_proc)
            env_proc.start()

    @property
    def observation_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    @property
    def done_space(self):
        return self._done_space

    @property
    def config(self):
        config_dict = {'global_observation_dim': self._num_agent * self._state_space.shape[-1]}
        return config_dict

    def step(self, action):
        return self.step_ops(action)

    def reset(self):
        return self.reset_ops()[0]

    def close(self):
        for env in self._envs:
            env.close()
        for env_proc in self.mpe_env_procs:
            env_proc.terminate()
            env_proc.join()
        return True

    def _step(self, actions):
        """Inner step function"""
        accum_env_num = 0
        for i in range(self._proc_num):
            env_num = self.mpe_env_procs[i].env_num
            self.action_queues[i].put(actions[accum_env_num: accum_env_num + env_num, ])
            accum_env_num += env_num
        results = []
        for i in range(self._proc_num):
            result = self.exp_queues[i].get()
            results.extend(result)
        local_obs, rewards, dones, _ = map(np.array, zip(*results))
        if dones.all():
            local_obs = self._reset()
        local_obs = local_obs.astype(np.float32)
        rewards = rewards.astype(np.float32)
        return local_obs, rewards, dones

    def _reset(self):
        """Inner reset function"""
        s0 = []
        for i in range(self._proc_num):
            self.action_queues[i].put('reset')
        for i in range(self._proc_num):
            s0.extend(self.init_state_queues[i].get())
        s0 = np.array(s0, np.float32)
        return s0

    def _space_adapter(self, gym_space, batch_shape=None):
        """Inner space adapter"""
        shape = gym_space.shape
        # The dtype get from gym.space is np.int64, but step() accept np.int32 actually.
        dtype = np.int32 if gym_space.dtype.type == np.int64 else gym_space.dtype.type
        if isinstance(gym_space, spaces.Discrete):
            return Space(shape, dtype, low=0, high=gym_space.n, batch_shape=batch_shape)

        return Space(shape, dtype, low=gym_space.low, high=gym_space.high, batch_shape=batch_shape)








