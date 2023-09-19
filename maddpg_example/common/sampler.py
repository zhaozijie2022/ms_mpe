from typing import Dict, List, Tuple
import torch
import numpy as np
import mindspore as ms

epsilon = 0.05


def rollout(env, agents, max_step) -> List[Dict[str, np.ndarray]]:
    """获得1条轨迹"""
    n = env.n
    obs_dim = agents[0].obs_dim
    action_dim = agents[0].action_dim
    max_step += 1

    traj_obs_n = [[] for _ in range(n)]
    traj_action_n = [[] for _ in range(n)]
    traj_reward_n = [[] for _ in range(n)]
    traj_obs_next_n = [[] for _ in range(n)]
    traj_done_n = [[] for _ in range(n)]

    obs_n = env.reset()
    action_n = [np.zeros((action_dim,)) for _ in range(n)]
    done_n = [False for _ in range(n)]
    cur_step = 0

    while not (all(done_n) or cur_step >= max_step):
        for i, agent in enumerate(agents):
            if np.random.uniform() < epsilon:
                action_n[i] = np.random.uniform(-1, 1, size=(action_dim,))
            else:
                action = agent.get_action(obs=obs_n[i])
                action_n[i] = action.asnumpy().squeeze()

        obs_next_n, reward_n, done_n, info_n = env.step(action_n)
        reward_n = [np.array(reward) for reward in reward_n]  # list of float -> list of np.ndarray

        for i in range(n):
            traj_obs_n[i].append(obs_n[i])
            traj_action_n[i].append(action_n[i])
            traj_reward_n[i].append(reward_n[i])
            traj_obs_next_n[i].append(obs_next_n[i])
            traj_done_n[i].append(done_n[i])

        cur_step += 1
        obs_n = obs_next_n[:]

    traj_n = []
    for i in range(n):
        traj_n.append(dict(
            obs=np.array(traj_obs_n[i][1:]),
            action=np.array(traj_action_n[i][1:]),
            reward=np.array(traj_reward_n[i][1:]),
            obs_next=np.array(traj_obs_next_n[i][1:]),
            done=np.array(traj_done_n[i][1:]),
        ))
    return traj_n


def sample_batch(agents, batch_size):
    buffer_size = agents[0].buffer.size
    # 理论上, agents.buffer[same_task_id]._size应该相等
    indices = np.random.randint(0, buffer_size, batch_size)
    samples_n = []
    for agent in agents:
        samples_n.append(agent.buffer.sample_batch(indices))
    # 返回list of dict, 类型和obtain_samples返回的trajs_n相同
    return samples_n


def evaluate(env, agents, max_step):
    """获得1条轨迹"""
    n = env.n
    action_dim = agents[0].action_dim
    max_step += 1

    obs_n = env.reset()
    action_n = [np.zeros((action_dim,)) for _ in range(n)]
    done_n = [False for _ in range(n)]
    cur_step = 0

    accumulated_reward = 0

    while not (all(done_n) or cur_step >= max_step):
        for i, agent in enumerate(agents):
            action = agent.get_action(obs=obs_n[i])
            action_n[i] = action.asnumpy().squeeze()

        obs_next_n, reward_n, done_n, info_n = env.step(action_n)

        cur_step += 1
        obs_n = obs_next_n[:]
        accumulated_reward += sum(reward_n)

    return cur_step, accumulated_reward / env.n



