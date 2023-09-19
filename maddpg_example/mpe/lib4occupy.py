import torch
import numpy as np
import time
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 因为使用了imp, 在python3.4中就被废弃了, 隐藏warning


def make_env(scenario_name, num=3, is_wind=False):
    """环境部分"""
    from mpe.environment import MultiAgentEnv
    import mpe.scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(num=num, is_wind=is_wind)
    # create multiagent environment
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,
                        done_callback=scenario.done)
    return env
















