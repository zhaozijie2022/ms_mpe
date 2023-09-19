from typing import List
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
import numpy as np

import pickle
import os
import copy

from common.utils import create_dir
from common.networks import MsMLP as MLP
from common.buffer import ReplayBuffer
from common.sampler import rollout, sample_batch


class Actor(MLP):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(Actor, self).__init__(input_dim, output_dim, hidden_dim)

    def forward(self, obs):
        return nn.Tanh()(super().forward(obs))


def ms_cat(mylist: List[np.ndarray], dim=1):
    ms_tensors = [ms.Tensor(arr) for arr in mylist]
    return ops.Concat(dim)(ms_tensors)


class MADDPGAgent:
    def __init__(
            self, n_agents, agent_id,
            obs_dim, action_dim,
            hidden_dim_act=64, hidden_dim_critic=64,
            max_buffer_size=int(5e5),
            gamma=0.99,
            actor_lr=3e-4, critic_lr=3e-4,
            batch_size=256, tau=0.01,
    ):
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim_act = hidden_dim_act
        self.hidden_dim_critic = hidden_dim_critic
        self.gamma = gamma
        self.max_buffer_size = max_buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.tau = tau
        self._train_epochs = 0  # 截止现在, 训练了多少次

        self.actor = Actor(input_dim=obs_dim,
                           output_dim=action_dim,
                           hidden_dim=hidden_dim_act)
        self.critic = MLP(input_dim=(obs_dim + action_dim) * n_agents,
                          output_dim=1,
                          hidden_dim=hidden_dim_critic)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        self.actor_optimizer = nn.Adam(self.actor.trainable_params(), learning_rate=actor_lr)
        self.critic_optimizer = nn.Adam(self.critic.trainable_params(), learning_rate=critic_lr)

        self.buffer = ReplayBuffer(obs_dim=obs_dim,
                                   action_dim=action_dim,
                                   max_size=max_buffer_size)

        self.mse_loss = nn.MSELoss()

    def soft_update_target_networks(self):
        for p, p_target in zip(self.actor.trainable_params(), self.target_actor.trainable_params()):
            p_target.set_data(p_target * (1 - self.tau) + p * self.tau)
        for p, p_target in zip(self.critic.trainable_params(), self.target_critic.trainable_params()):
            p_target.set_data(p_target * (1 - self.tau) + p * self.tau)

    def get_action(self, obs: np.ndarray) -> ms.Tensor:
        obs = ms.Tensor(obs, dtype=ms.float32).view(-1, self.obs_dim)
        action = self.actor.forward(obs)
        return action

    def forward_fn_critic(self, data, label):
        obs_n, action_n = data
        q_value = self.critic.forward(ops.Concat(1)([ms_cat(obs_n, dim=1),
                                                     ms_cat(action_n, dim=1)]))
        critic_loss = self.mse_loss(q_value, label)
        return critic_loss, label  # q_value is logits

    def forward_fn_actor(self, data, label=None):
        obs_n, action_n = data
        action_n[0] = self.actor.forward(obs_n[0])
        actor_loss = - self.critic.forward(ops.Concat(1)([ms_cat(obs_n, dim=1),
                                                          ms_cat(action_n, dim=1)])).mean()
        return actor_loss, label

    def train_step_critic(self, data, label):
        grad_fn = ops.value_and_grad(self.forward_fn_critic,
                                     None,
                                     self.critic_optimizer.parameters,
                                     has_aux=False)
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, self.critic_optimizer(grads))
        return loss

    def train_step_actor(self, data):
        grad_fn = ops.value_and_grad(self.forward_fn_actor,
                                     None,
                                     self.actor_optimizer.parameters,
                                     has_aux=True)
        (loss, _), grads = grad_fn(data)
        loss = ops.depend(loss, self.actor_optimizer(grads))
        return loss

    def train(self, agents):
        if self.buffer.size < self.batch_size:
            return

        self._train_epochs += 1
        trans_n = sample_batch(agents=agents, batch_size=self.batch_size)

        for trans in trans_n:
            for key in trans.keys():
                trans[key] = ms.Tensor(trans[key], dtype=ms.float32)

        obs_next_n = [trans_n[j]['obs_next'] for j in range(self.n_agents)]
        action_next_n = [agents[j].target_actor.forward(obs_next_n[j]) for j in range(self.n_agents)]
        obs_next_n.insert(0, obs_next_n.pop(self.agent_id))
        action_next_n.insert(0, action_next_n.pop(self.agent_id))
        q_next = self.target_critic.forward(ops.Concat(1)([ms_cat(obs_next_n, dim=1),
                                                           ms_cat(action_next_n, dim=1)]))
        tmp = trans_n[self.agent_id]['reward'] + \
                   (1 - trans_n[self.agent_id]['done']) * self.gamma * q_next
        target_q = ms.Tensor(tmp.asnumpy(), dtype=ms.float32)

        obs_n = [trans_n[j]["obs"] for j in range(self.n_agents)]
        action_n = [trans_n[j]["action"] for j in range(self.n_agents)]
        obs_n.insert(0, obs_n.pop(self.agent_id))
        action_n.insert(0, action_n.pop(self.agent_id))

        self.train_step_critic(data=[obs_n, action_n], label=target_q)
        self.train_step_actor(data=[obs_n, action_n])

        self.soft_update_target_networks()

    def save_model(self, save_path):
        save_path = os.path.join(save_path, "agent_%d" % self.agent_id)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        model_dict = {"actor": self.actor, "critic": self.critic,}
        for key in model_dict.keys():
            model_save_path = os.path.join(save_path, key + ".ckpt")
            ms.save_checkpoint(model_dict[key], model_save_path)

    def load_model(self, load_path):
        load_path = os.path.join(load_path, "agent_%d" % self.agent_id)
        model_dict = {"actor": self.actor, "critic": self.critic,}
        for key in model_dict.keys():
            model_load_path = os.path.join(load_path, key + ".ckpt")
            param_dict = ms.load_checkpoint(model_load_path)
            ms.load_param_into_net(model_dict[key], param_dict)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

    def save_buffer(self, save_path):
        save_path = os.path.join(save_path, "buffer_" + "agent_%d" % self.agent_id + ".pkl")
        with open(save_path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, load_path):
        load_path = os.path.join(load_path, "buffer_" + "agent_%d" % self.agent_id + ".pkl")
        with open(load_path, "rb") as f:
            self.buffer = pickle.load(f)








