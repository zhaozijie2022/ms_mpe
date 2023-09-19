import os
import time
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.spaces import Box, Discrete

from common.utils import create_dir
from mpe.lib4occupy import make_env
# from marl.maddpg import MADDPGAgent
from marl.maddpg_ms import MADDPGAgent
from common.sampler import rollout, sample_batch


num_agents = 3
scenario_name = "occupy"
hidden_dim_act = 32
hidden_dim_critic = 64
device = torch.device(torch.device("cpu"))
max_buffer_size = 100000
max_step = 50 + 1

load_models_path = "./models/"
if __name__ == '__main__':
    env = make_env(scenario_name=scenario_name, num=num_agents)
    obs_dim = env.observation_space[0].shape[0]  # obs_dim: 2 * 3 + 2 = 8
    if isinstance(env.action_space[0], Discrete):
        action_dim = env.action_space[0].n
    else:
        action_dim = env.action_space[0].shape[0]
    agents = [MADDPGAgent(n_agents=num_agents,
                          agent_id=i,
                          obs_dim=obs_dim,
                          action_dim=action_dim,
                          hidden_dim_act=hidden_dim_act,
                          hidden_dim_critic=hidden_dim_critic,
                          max_buffer_size=max_buffer_size,
                          # device=device,
                          ) for i in range(num_agents)]
    for agent in agents:
        agent.load_model(load_models_path)

    n = env.n
    while True:
        obs_n = env.reset()
        action_n = [np.zeros((action_dim,)) for _ in range(n)]
        reward_n = [np.zeros((1,)) for _ in range(n)]

        obs_next_n = [np.zeros((obs_dim,)) for _ in range(n)]
        # endregion

        cur_step = 0
        while True:
            # region S1.1 与环境交互, 将数据存储到xxx_n中
            with torch.no_grad():
                for i, agent in enumerate(agents):
                    action = agent.get_action(obs=obs_n[i])
                    # action_n[i] = action.cpu().numpy().squeeze()
                    action_n[i] = action.asnumpy().squeeze()

            time.sleep(0.05)
            env.render()
            obs_next_n, reward_n, done_n, info_n = env.step(action_n)
            # print(done_n)
            reward_n = [np.array(reward) for reward in reward_n]  # list of float -> list of np.ndarray

            cur_step += 1
            obs_n = obs_next_n[:]

            if (cur_step == max_step) or all(done_n):
                obs_n = env.reset()
                action_n = [np.zeros((action_dim,)) for _ in range(n)]
                reward_n = [np.zeros((1,)) for _ in range(n)]
                obs_next_n = [np.zeros((obs_dim,)) for _ in range(n)]
                cur_step = 0



