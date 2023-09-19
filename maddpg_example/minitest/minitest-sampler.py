# # the test code for pearl.sampler.py
#
# from mpe.lib4occupy import *
# from pearl.networks import MLP, VaeEncoder
# from pearl.marl.maddpg import MADDPGAgent
# from pearl.sampler import Sampler
# import numpy as np
# import torch
#
# # S1. 初始化参数
# num_train_tasks = 1
# num_agents = 3
# num_landmarks = 3
# latent_dim = 5  # z的维度
# hidden_dim_act = 8
# hidden_dim_critic = 8
# hidden_dim_en = 8
# max_step = 200  # 每个episode的最大step数
# device = torch.device(torch.device("cuda"))
# max_samples = 200  # 采样的最大样本数
#
#
# # S2. 初始化envs, agents, sampler.py
# # region
# task_ids = list(range(num_train_tasks))
# task_infos = [np.random.uniform(-1, 1, size=(num_landmarks, 2)) for _ in range(num_train_tasks)]
# env = make_env(scenario_name="occupy", num=num_agents)
# obs_dim = env.observation_space[0].shape[0]  # obs_dim: 2 * 3 + 2 = 8
# act_dim = env.action_space[0].n
# agents = [MADDPGAgent(num_agents=num_agents,
#                       agent_id=i,
#                       obs_dim=obs_dim,
#                       action_dim=act_dim,
#                       latent_dim=latent_dim,
#                       hidden_dim_act=8,
#                       hidden_dim_critic=16,
#                       hidden_dim_en=8,
#                       max_buffer_size=10000,
#                       device=device) for i in range(num_agents)]
# sampler = Sampler(env=env, agents=agents, max_step=max_step, device=device)
# # endregion
#
# is_post_z = True
#
# # region sampler.rollout
# # traj_obs_past_n = [[] for _ in range(env.n)]
# # traj_action_past_n = [[] for _ in range(env.n)]
# # traj_reward_past_n = [[] for _ in range(env.n)]
# # traj_hidden_past_n = [[] for _ in range(env.n)]
# #
# # traj_obs_n = [[] for _ in range(env.n)]
# # traj_action_n = [[] for _ in range(env.n)]
# # traj_reward_n = [[] for _ in range(env.n)]
# # traj_hidden_n = [[] for _ in range(env.n)]
# #
# # traj_obs_next_n = [[] for _ in range(env.n)]
# # traj_done_n = [[] for _ in range(env.n)]
# #
# # obs_past_n = [np.zeros(obs_dim) for _ in range(env.n)]
# # action_past_n = [np.zeros(act_dim) for _ in range(env.n)]
# # reward_past_n = [0 for _ in range(env.n)]
# # hidden_past_n = [np.zeros(hidden_dim_en) for _ in range(env.n)]
# #
# # obs_n = env.reset()
# # action_n = [np.zeros(act_dim) for _ in range(env.n)]
# # reward_n = [np.zeros(1) for _ in range(env.n)]
# # hidden_n = [np.zeros(hidden_dim_en) for _ in range(env.n)]
# #
# # done_n = [False for _ in range(env.n)]
# # obs_next_n = [np.zeros(obs_dim) for _ in range(env.n)]
# #
# # cur_step = 0
# # while not (all(done_n) or cur_step >= max_step):
# #     with torch.no_grad():
# #         for i in range(env.n):
# #             if cur_step == 0:
# #                 latent, hidden = agents[i].get_prior_latent(obs=obs_n[i],
# #                                                             hidden=hidden_n[i])
# #             else:
# #                 latent, hidden = agents[i].get_post_latent(obs_past_n[i],
# #                                                            action_past_n[i],
# #                                                            reward_past_n[i],
# #                                                            hidden_past_n[i])
# #             action = agents[i].get_action(obs=obs_n[i], latent=latent)
# #             action_n[i] = action.cpu().numpy().squeeze()
# #             hidden_n[i] = hidden.squeeze().cpu().numpy()
# #
# #         obs_next_n, reward_n, done_n, _ = env.step(action_n)
# #         reward_n = [np.array(reward) for reward in reward_n]
# #
# #         for i in range(env.n):
# #             traj_obs_past_n[i].append(obs_past_n[i])
# #             traj_action_past_n[i].append(action_past_n[i])
# #             traj_reward_past_n[i].append(reward_past_n[i])
# #             traj_hidden_past_n[i].append(hidden_past_n[i])
# #
# #             traj_obs_n[i].append(obs_n[i])
# #             traj_action_n[i].append(action_n[i])
# #             traj_reward_n[i].append(reward_n[i])
# #             traj_hidden_n[i].append(hidden_n[i])
# #
# #             traj_obs_next_n[i].append(obs_next_n[i])
# #             traj_done_n[i].append(done_n[i])
# #
# #         cur_step += 1
# #         obs_past_n = obs_n
# #         action_past_n = action_n
# #         reward_past_n = reward_n
# #         hidden_past_n = hidden_n
# #
# #     traj_n = []
# #     for i in range(env.n):
# #         traj_n.append(dict(
# #             obs_past=traj_obs_past_n[i][1:],
# #             action_past=traj_action_past_n[i][1:],
# #             reward_past=traj_reward_past_n[i][1:],
# #             hidden_past=traj_hidden_past_n[i][1:],
# #
# #             obs=traj_obs_n[i][1:],
# #             action=traj_action_n[i][1:],
# #             reward=traj_reward_n[i][1:],
# #             hidden=traj_hidden_n[i][1:],
# #
# #             obs_next=traj_obs_next_n[i],
# #             done=traj_done_n[i]
# #         ))
# # endregion
#
# num_samples, trajs_n = sampler.obtain_samples(max_samples=max_samples,
#                                               is_post_z=is_post_z)
#
# for i in range(env.n):
#     agents[i].buffer.add_traj(trajs_n[i])
#
# for i in range(env.n):
#     print(agents[i].buffer._size)
#
# # 从buffer中采样
#
# trans_n = sampler.sample_batch(batch_size=10)
#
#
#
# print(1)



from typing import Dict, List, Tuple
import torch
import numpy as np
from pearl.marl.maddpg import MADDPGAgent

# def rollout(env, agents, max_step) -> List[Dict[str, np.ndarray]]:
#     """获得1条轨迹"""
#     n = env.n
#     obs_dim = agents[0].obs_dim
#     action_dim = agents[0].action_dim
#     hidden_dim_en = agents[0].hidden_dim_en
#     max_step += 1
#     # region S0.1 初始化traj_xxx_n: List[List[np.ndarray]] (用于存储历史数据)
#     traj_obs_past_n = [[] for _ in range(n)]
#     traj_action_past_n = [[] for _ in range(n)]
#     traj_reward_past_n = [[] for _ in range(n)]
#     traj_hidden_past_n = [[] for _ in range(n)]
#
#     traj_obs_n = [[] for _ in range(n)]
#     traj_action_n = [[] for _ in range(n)]
#     traj_hidden_n = [[] for _ in range(n)]
#     traj_reward_n = [[] for _ in range(n)]
#
#     traj_obs_next_n = [[] for _ in range(n)]
#     traj_done_n = [[] for _ in range(n)]
#     # endregion
#
#     # region S0.2 初始化xxx_n: List[np.ndarray] (用于存储当前时刻的数据)
#     obs_past_n = [np.zeros((obs_dim,)) for _ in range(n)]
#     action_past_n = [np.zeros((action_dim,)) for _ in range(n)]
#     reward_past_n = [np.zeros((1,)) for _ in range(n)]
#     hidden_past_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]
#
#     obs_n = env.reset()
#     action_n = [np.zeros((action_dim,)) for _ in range(n)]
#     reward_n = [np.zeros((1,)) for _ in range(n)]
#     hidden_n = [np.zeros((hidden_dim_en,)) for _ in range(n)]
#
#     obs_next_n = [np.zeros((obs_dim,)) for _ in range(n)]
#     done_n = [False for _ in range(n)]
#     # endregion
#
#     cur_step = 0
#     while not (all(done_n) or cur_step >= max_step):
#         # region S1.1 与环境交互, 将数据存储到xxx_n中
#         with torch.no_grad():
#             for i, agent in enumerate(agents):
#                 if cur_step == 0:
#                     latent, hidden = agent.get_prior_latent(obs=obs_n[i],
#                                                             hidden=hidden_n[i])
#                 else:
#                     latent, hidden = agent.get_post_latent(obs=obs_past_n[i],
#                                                            action=action_past_n[i],
#                                                            reward=reward_past_n[i],
#                                                            hidden=hidden_past_n[i])
#                 # latent = latent.cpu().numpy().squeeze()  # (1, latent_dim) -> (latent_dim,)
#                 action = agent.get_action(obs=obs_n[i], latent=latent)
#                 action_n[i] = action.cpu().numpy().squeeze()  # (1, action_dim) -> (action_dim,)
#                 hidden_n[i] = hidden.cpu().numpy().squeeze()  # (1, hidden_dim) -> (hidden_dim,)
#
#         obs_next_n, reward_n, done_n, info_n = env.step(action_n)
#         reward_n = [np.array(reward) for reward in reward_n]  # list of float -> list of np.ndarray
#         # endregion
#
#         # region S1.2 将当前时刻的数据存储到traj_xxx_n中
#         for i in range(n):
#             traj_obs_past_n[i].append(obs_past_n[i])
#             traj_action_past_n[i].append(action_past_n[i])
#             traj_reward_past_n[i].append(reward_past_n[i])
#             traj_hidden_past_n[i].append(hidden_past_n[i])
#
#             traj_obs_n[i].append(obs_n[i])
#             traj_action_n[i].append(action_n[i])
#             traj_reward_n[i].append(reward_n[i])
#             traj_hidden_n[i].append(hidden_n[i])
#
#             traj_obs_next_n[i].append(obs_next_n[i])
#             traj_done_n[i].append(done_n[i])
#         # endregion
#
#         # region S1.3 更新xxx_n与xxx_past_n
#         cur_step += 1
#         obs_past_n = obs_n
#         action_past_n = action_n
#         reward_past_n = reward_n
#         hidden_past_n = hidden_n
#         # endregion
#         # print(cur_step)
#
#     traj_n = []
#     for i in range(n):
#         traj_n.append(dict(
#             obs_past=np.array(traj_obs_past_n[i][1:]),
#             action_past=np.array(traj_action_past_n[i][1:]),
#             reward_past=np.array(traj_reward_past_n[i][1:]),
#             hidden_past=np.array(traj_hidden_past_n[i][1:]),
#
#             obs=np.array(traj_obs_n[i]),
#             action=np.array(traj_action_n[i]),
#             reward=np.array(traj_reward_n[i]),
#             hidden=np.array(traj_hidden_n[i]),
#
#             obs_next=np.array(traj_obs_next_n[i]),
#             done=np.array(traj_done_n[i]),
#         ))
#     return traj_n


