import numpy as np
from mpe.core import World


class WindWorld(World):
    """随机吹风, 环境会在所有agent身上施加一个随机大小和方向的风力,
    每次初始化时随机生成, 以模拟environment dynamic的变化"""
    def __init__(self, is_wind=False):
        super(WindWorld, self).__init__()
        self.is_wind = is_wind
        if self.is_wind:
            self.wind_force = np.zeros(self.dim_p)  # 风力

    def step(self):
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        p_force = [None for _ in range(len(self.entities))]
        p_force = super(WindWorld, self).apply_action_force(p_force)
        p_force = super(WindWorld, self).apply_environment_force(p_force)
        if self.is_wind:
            p_force = self.apply_wind_force(p_force)

        super(WindWorld, self).integrate_state(p_force)

    def apply_wind_force(self, p_force):
        for i, agent in enumerate(self.agents):
            if agent.movable:
                if p_force[i] is None:
                    p_force[i] = 0.0
                p_force[i] += self.wind_force
        return p_force






