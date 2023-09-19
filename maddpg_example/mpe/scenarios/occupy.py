# scenario: occupy, 用于将pearl的单智能体多任务算法应用到mpe中
# n个agent去追逐n个landmark
# done: agents和与他最近的landmark中心距离均0.05时done
# reward: -1 * dist (agents与landmark的距离)
# obs: 自身位置 (landmark的位置不作为obs)
# info: 无


import numpy as np
from mpe.core import World, Agent, Landmark
# 无需新定义world，直接使用mpe.core.World
from mpe.scenarios.BaseScenario import BaseScenario
from mpe.WindWorld import WindWorld


class Scenario(BaseScenario):
    def make_world(self, num=3, is_wind=False):
        world = WindWorld(is_wind=is_wind)
        world.dim_c = 1
        num_agents = num
        num_landmarks = num
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.05
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # # pos_landmarks = np.array([[0.5, 0.5], [-0.5, -0.5]])
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = pos_landmarks[i]
        #     landmark.state.p_vel = np.zeros(world.dim_p)
        if world.is_wind:
            tmp = np.random.uniform(-1, +1, world.dim_p)
            world.wind_force = tmp/np.linalg.norm(tmp) * 0.1

    def reward(self, agent, world):
        # 返回这个agent的reward, 寻找与它最近的landmark
        rew = 0.0
        for landmark in world.landmarks:
            dists = [np.linalg.norm(ag.state.p_pos - landmark.state.p_pos) for ag in world.agents]
            rew -= min(dists)
        # 增加出界惩罚
        # bound = 1.85
        # if agent.state.p_pos[0] < -bound or \
        #         agent.state.p_pos[0] > bound or \
        #         agent.state.p_pos[1] < -bound or \
        #         agent.state.p_pos[1] > bound:
        #     return -1000
        return rew

    def observation(self, agent, world):
        # 可见其他agent的位置, 自身速度与位置, landmark不可见
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        for landmark in world.landmarks:
            other_pos.append(landmark.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos)
    
    def done(self, agent, world):
        # 判断是否出界
        # bound = 2
        # for ag in world.agents:
        #     if ag.state.p_pos[0] < -bound or \
        #             ag.state.p_pos[0] > bound or \
        #             ag.state.p_pos[1] < -bound or \
        #             ag.state.p_pos[1] > bound:
        #         return True
        # dist = 0.0
        # for landmark in world.landmarks:
        #     dists = [np.linalg.norm(ag.state.p_pos - landmark.state.p_pos) for ag in world.agents]
        #     dist += min(dists)
        # if dist < 0.05 * len(world.agents):
        #     return True
        # else:
        #     return False
        return False
