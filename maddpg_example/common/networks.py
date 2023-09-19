
# 数据输入输出networks.py都是tensor.float32, 不设计和numpy的转换
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mindspore.nn as nn_ms
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int = 2,
        hidden_activation: torch.nn.functional = F.relu,
        init_w: float = 3e-3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_activation = hidden_activation

        # Set fully connected layers
        self.fc_layers = nn.ModuleList()
        self.hidden_layers = [hidden_dim] * num_hidden_layers  # 此时的hidden_layers只是一个int列表
        in_layer = input_dim

        for i, hidden_layer in enumerate(self.hidden_layers):
            fc_layer = nn.Linear(in_layer, hidden_layer)
            in_layer = hidden_layer  # 上一层的输出维度就是下一层的输入维度
            self.__setattr__("fc_layer{}".format(i), fc_layer)  # 创建一个名为fc_layeri的属性, 其值为nn.Linear
            self.fc_layers.append(fc_layer)  # 此时的self.fc_layers已装填好nn.Linear

        # 定义输出层
        self.last_fc_layer = nn.Linear(hidden_dim, output_dim)
        self.last_fc_layer.weight.data.uniform_(-init_w, init_w)
        self.last_fc_layer.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        for fc_layer in self.fc_layers:
            x = self.hidden_activation(fc_layer(x))
        x = self.last_fc_layer(x)
        return x


class MsMLP(nn_ms.Cell):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=64,
            hidden_activation=nn_ms.ReLU(),
    ):
        super(MsMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_activation = hidden_activation

        self.fc1 = nn_ms.Dense(input_dim, hidden_dim)
        self.fc2 = nn_ms.Dense(hidden_dim, hidden_dim)
        self.fc3 = nn_ms.Dense(hidden_dim, output_dim)

    def forward(self, x):
        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.fc3(x)
        return x























