# the test code for pearl.train.py

from mpe.lib4occupy import make_env
from marl.maddpg import MADDPGAgent
from common.sampler import rollout, sample_batch
from torchviz import make_dot
from common.arguments import get_args

import numpy as np
import torch
import torch.nn.functional as F

# region S1. 初始化参数
args = get_args(
    env_name="mpe",
    scenario_name="occupy",
    is_wind=True,
    num_agents=2,

    latent_dim=16,
    hidden_dim_act=32,
    hidden_dim_critic=64,
    hidden_dim_en=32,
    max_buffer_size=100000,  # 100k
    actor_lr=1e-4,
    critic_lr=1e-3,
    encoder_lr=5e-4,
    kl_lambda=0.1,

    max_step=100,
    is_train=True,
    is_display=False,
    load_models_path=None,
    load_buffers_path=None,

    gamma=0.99,
    batch_size=512,
    num_episodes=60000,
    train_rate=2,
    print_rate=100,
    save_rate=5000,
    save_buffer_rate=10000,

    save_models_path=None,
    save_figures_path=None,
    save_buffers_path=None,

    device="cuda" if torch.cuda.is_available() else "cpu",
)
# endregion


# region S2. 初始化env, agents, sampler
env = make_env(scenario_name=args.scenario_name,
               num=args.num_agents,
               is_wind=args.is_wind)
obs_dim = env.observation_space[0].shape[0]  # obs_dim: 2 * 3 + 2 = 8
action_dim = env.action_space[0].n
agents = [MADDPGAgent(n_agents=args.num_agents,
                      agent_id=agent_id,
                      obs_dim=obs_dim,
                      action_dim=action_dim,
                      latent_dim=args.latent_dim,
                      hidden_dim_act=args.hidden_dim_act,
                      hidden_dim_critic=args.hidden_dim_critic,
                      hidden_dim_en=args.hidden_dim_en,
                      max_buffer_size=args.max_buffer_size,
                      device=torch.device(args.device),
                      gamma=args.gamma,
                      actor_lr=args.actor_lr,
                      critic_lr=args.critic_lr,
                      encoder_lr=args.encoder_lr,
                      kl_lambda=args.kl_lambda,
                      batch_size=args.batch_size,
                      ) for agent_id in range(args.num_agents)]

# endregion

# region S3. 采样max_samples个样本并压入buffer
samples_n = rollout(env, agents, max_step=args.max_step)

for i in range(env.n):
    agents[i].buffer.add_traj(samples_n[i])

for i in range(env.n):
    print("agent%d: " % i + str(agents[i].buffer.size) + '/' + str(agents[i].buffer.max_size))
# endregion

# region S4. 从buffer中采样batch_size个样本, 并转化为tensor
trans_n = sample_batch(agents, batch_size=16)
# trans_n: List[Dict[str, np.ndarray]] np.ndarray.shape = (batch_size, obs_dim)

for trans in trans_n:
    for key in trans.keys():
        trans[key] = torch.tensor(trans[key], dtype=torch.float32, device=args.device)
        # trans[key].shape = (batch_size, xxx_dim or 1)
# endregion

agent_id = 0
self = agents[agent_id]
self.batch_size = 16
# region S1. kl_loss
kl_div, latent = self.task_encoder.get_kl_div(
    obs=trans_n[self.agent_id]["obs_past"],
    action=trans_n[self.agent_id]["action_past"],
    reward=trans_n[self.agent_id]["reward_past"],
    hidden=trans_n[self.agent_id]["hidden_past"])
target_latent, _ = self.target_task_encoder.get_latent(
    obs_past=trans_n[self.agent_id]["obs_past"],
    action_past=trans_n[self.agent_id]["action_past"],
    reward_past=trans_n[self.agent_id]["reward_past"],
    hidden_past=trans_n[self.agent_id]["hidden_past"])
kl_loss = kl_div * self.kl_lambda
# endregion

# region S2. critic_loss
with torch.no_grad():
    obs_next_n = [trans_n[j]["obs_next"] for j in range(self.num_agents)]
    target_latent_next_n = [torch.zeros(self.batch_size, self.latent_dim, device=self.device) for _ in
                            range(self.num_agents)]
    action_next_n = [torch.zeros(self.batch_size, self.action_dim, device=self.device) for _ in range(self.num_agents)]
    for j, agent_j in enumerate(agents):
        target_latent_next_n[j], _ = agent_j.target_task_encoder.get_latent(
            obs_past=trans_n[j]["obs"],
            action_past=trans_n[j]["action"],
            reward_past=trans_n[j]["reward"],
            hidden_past=trans_n[j]["hidden"],
        )
        action_next_n[j] = agent_j.target_actor.forward(torch.cat([obs_next_n[j],
                                                                   target_latent_next_n[j]], dim=1))
    obs_next_n.insert(0, obs_next_n.pop(self.agent_id))
    action_next_n.insert(0, action_next_n.pop(self.agent_id))

    q_next = self.target_critic.forward(torch.cat([torch.cat(obs_next_n, dim=1),
                                                   torch.cat(action_next_n, dim=1),
                                                   target_latent_next_n[self.agent_id]], dim=1)).detach()
    target_q = (trans_n[self.agent_id]["reward"] +
                (1 - trans_n[self.agent_id]["done"]) * self.gamma * q_next).detach()

obs_n = [trans_n[j]["obs"] for j in range(self.num_agents)]
obs_n.insert(0, obs_n.pop(self.agent_id))

action_n = [trans_n[j]["action"] for j in range(self.num_agents)]
action_n.insert(0, action_n.pop(self.agent_id))

q_value = self.critic.forward(torch.cat([torch.cat(obs_n, dim=1),
                                         torch.cat(action_n, dim=1),
                                         latent], dim=1))
critic_loss = F.mse_loss(q_value, target_q)
# endregion

# region S3. actor_loss
action_n[0] = self.actor.forward(torch.cat([obs_n[0], latent], dim=1))
actor_loss = -self.critic.forward(torch.cat([torch.cat(obs_n, dim=1),
                                             torch.cat(action_n, dim=1),
                                             latent], dim=1)).mean()
# endregion

# region S4. 梯度反传, 同步target网络
total_loss = critic_loss + actor_loss + kl_loss
self.actor_optimizer.zero_grad()
self.critic_optimizer.zero_grad()
self.task_encoder_optimizer.zero_grad()
total_loss.backward()
self.actor_optimizer.step()
self.critic_optimizer.step()
self.task_encoder_optimizer.step()

self.soft_update_target_networks(self.tau)





