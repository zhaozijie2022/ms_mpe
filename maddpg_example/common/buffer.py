#  采样transition [obs, action, reward, obs_next, done
from typing import Dict, List, Tuple, Any
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size):
        self._obs = np.zeros((max_size, obs_dim))
        self._action = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        # self._hidden = np.zeros((max_size, hidden_dim_en))

        self._obs_next = np.zeros((max_size, obs_dim))
        self._done = np.zeros((max_size, 1))

        self.max_size = max_size
        self._top = 0
        self.size = 0

    def clear(self):
        self._top = 0  # 指针, 指向下一个要写入的位置
        self.size = 0  # 当前buffer大小

    def add(self, obs, action, reward, obs_next, done):
        # 压入一个transition, 输入 np.ndarray
        self._obs[self._top] = obs
        self._action[self._top] = action
        self._reward[self._top] = reward
        self._obs_next[self._top] = obs_next
        self._done[self._top] = done

        self._top = (self._top + 1) % self.max_size  # 满了就从头开始覆盖
        if self.size < self.max_size:
            self.size += 1

    def add_traj(self, traj: Dict[str, np.ndarray]):
        for (obs, action, reward, obs_next, done) in zip(
                traj["obs"], traj["action"], traj["reward"],
                traj["obs_next"], traj["done"],
        ):
            self.add(obs, action, reward, obs_next, done)

    def sample_batch(self, indices):
        # 给定indices, 采样transition
        return dict(
            obs=self._obs[indices],
            action=self._action[indices],
            reward=self._reward[indices],

            obs_next=self._obs_next[indices],
            done=self._done[indices],
        )
