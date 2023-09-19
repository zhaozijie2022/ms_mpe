import numpy as np
import torch
import os
import time
from mpe.lib4occupy import make_env
from typing import List
from itertools import permutations


scenario_name = "occupy"
max_ep_len = 100
num_eps = 10


def generate_pairs(landmark_pos: List[np.ndarray], agent_pos: List[np.ndarray]) -> List[List[np.ndarray]]:
    # 计算所有可能的地标和机器人的排列组合
    all_pairs = list(permutations(range(len(landmark_pos)), len(agent_pos)))

    # 初始化最小总距离和对应的最优组合
    min_distance = float('inf')
    best_pairs = None

    # 遍历所有排列组合
    for pairs in all_pairs:
        # 计算当前组合的总距离
        total_distance = sum(
            np.linalg.norm(landmark_pos[landmark_idx] - agent_pos[agent_idx]) for landmark_idx, agent_idx in
            enumerate(pairs))

        # 更新最小总距离和最优组合
        if total_distance < min_distance:
            min_distance = total_distance
            best_pairs = pairs

    # 根据最优组合生成对应的坐标对
    result = [[landmark_pos[landmark_idx], agent_pos[agent_idx]] for landmark_idx, agent_idx in enumerate(best_pairs)]

    return result


def get_actions(landmark_pos: List[np.ndarray], agent_pos: List[np.ndarray]) -> List[np.ndarray]:
    directions = []

    # 生成坐标对
    pairs = generate_pairs(landmark_pos, agent_pos)

    for pair in pairs:
        landmark, agent = pair
        direction = landmark - agent
        # direction /= np.linalg.norm(direction)  # 归一化方向向量
        directions.append(direction)

    return directions


if __name__ == '__main__':
    env = make_env(scenario_name, num=3, is_wind=False)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    obs_n = env.reset()
    episode_step = 0
    num_ep = 0
    while True:
        # action_n = [env.action_space[i].sample() for i in range(env.n)]


        # action_n = []
        # for i in range(env.n):
        #     # tmp = np.random.uniform(0, 1, size=5)
        #     # tmp = np.exp(tmp) / np.sum(np.exp(tmp))
        #     for landmark in env.world.landmarks:
        #         dists = [np.linalg.norm(ag.state.p_pos - landmark.state.p_pos) for ag in env.world.agents]
        #     tmp = np.zeros(5)
        #     action_n.append(tmp)

        action_n = get_actions([landmark.state.p_pos for landmark in env.world.landmarks],
                               [ag.state.p_pos for ag in env.world.agents])
        time.sleep(0.5)
        env.render()
        new_obs_n, rew_n, done_n, _ = env.step(action_n)
        obs_n = new_obs_n
        episode_step += 1
        terminal = (episode_step >= max_ep_len) or all(done_n)
        if terminal or sum(rew_n) > -0.1:
            num_ep += 1
            print("Eps %d, Sps: %d" % (num_ep, episode_step))
            episode_step = 0
            obs_n = env.reset()
        if num_ep >= num_eps:
            break

















