"""神经网络模型定义"""

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络，用于PPO算法
    支持参数共享：同一物种的所有实体共享一个网络
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,  # [speed_delta, angular_velocity_delta]
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        activation: str = "tanh",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 激活函数选择
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU

        # 共享特征提取网络
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Actor网络（策略网络）
        # 输出动作的均值
        self.actor_mean = nn.Linear(prev_dim, action_dim)

        # 动作的标准差（可学习参数）
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic网络（价值网络）
        self.critic = nn.Linear(prev_dim, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        # Actor输出层使用较小的初始化
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        # Critic输出层
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观察 [batch_size, obs_dim]
            deterministic: 是否使用确定性策略

        Returns:
            action: 动作 [batch_size, action_dim]
            log_prob: 动作的对数概率 [batch_size]
            value: 状态价值 [batch_size]
            entropy: 熵 [batch_size]
        """
        # 共享特征
        features = self.shared_net(obs)

        # Actor: 计算动作分布
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)

        # 从高斯分布中采样动作
        if deterministic:
            action = action_mean
        else:
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()

        # 计算对数概率
        action_dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = action_dist.log_prob(action).sum(dim=-1)

        # 计算熵
        entropy = action_dist.entropy().sum(dim=-1)

        # Critic: 计算状态价值
        value = self.critic(features).squeeze(-1)

        return action, log_prob, value, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定动作的价值和概率（用于训练）

        Args:
            obs: 观察 [batch_size, obs_dim]
            actions: 动作 [batch_size, action_dim]

        Returns:
            log_prob: 动作的对数概率 [batch_size]
            value: 状态价值 [batch_size]
            entropy: 熵 [batch_size]
        """
        # 共享特征
        features = self.shared_net(obs)

        # Actor: 动作分布
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)
        action_dist = torch.distributions.Normal(action_mean, action_std)

        # 计算对数概率
        log_prob = action_dist.log_prob(actions).sum(dim=-1)

        # 计算熵
        entropy = action_dist.entropy().sum(dim=-1)

        # Critic: 状态价值
        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        获取状态价值

        Args:
            obs: 观察 [batch_size, obs_dim]

        Returns:
            value: 状态价值 [batch_size]
        """
        features = self.shared_net(obs)
        value = self.critic(features).squeeze(-1)
        return value


class SharedPolicy:
    """
    共享策略包装器
    每个物种使用一个网络，所有同物种的实体共享参数
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # 为每个物种创建一个网络
        self.hunter_net = ActorCriticNetwork(
            obs_dim, action_dim, hidden_dims
        ).to(self.device)

        self.prey_net = ActorCriticNetwork(
            obs_dim, action_dim, hidden_dims
        ).to(self.device)

    def get_network(self, entity_type: str) -> ActorCriticNetwork:
        """根据实体类型获取对应的网络"""
        if entity_type == "hunter":
            return self.hunter_net
        elif entity_type == "prey":
            return self.prey_net
        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

    def get_action(
        self, obs: np.ndarray, entity_type: str, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        获取动作

        Args:
            obs: 观察 [obs_dim]
            entity_type: 实体类型 ("hunter" 或 "prey")
            deterministic: 是否使用确定性策略

        Returns:
            action: 动作 [action_dim]
            log_prob: 动作的对数概率
            value: 状态价值
        """
        net = self.get_network(entity_type)
        net.eval()

        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob, value, _ = net(obs_tensor, deterministic)

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().item()
            value = value.cpu().item()

        return action, log_prob, value

    def get_actions_batch(
        self, obs_dict: dict, deterministic: bool = False
    ) -> Tuple[dict, dict, dict]:
        """
        批量获取动作（用于多智能体环境）

        Args:
            obs_dict: {entity_id: (obs, entity_type)}
            deterministic: 是否使用确定性策略

        Returns:
            actions_dict: {entity_id: action}
            log_probs_dict: {entity_id: log_prob}
            values_dict: {entity_id: value}
        """
        actions_dict = {}
        log_probs_dict = {}
        values_dict = {}

        # 按物种分组
        hunter_obs = []
        hunter_ids = []
        prey_obs = []
        prey_ids = []

        for entity_id, (obs, entity_type) in obs_dict.items():
            if entity_type == "hunter":
                hunter_obs.append(obs)
                hunter_ids.append(entity_id)
            else:
                prey_obs.append(obs)
                prey_ids.append(entity_id)

        # 批量处理猎人
        if hunter_obs:
            hunter_obs_tensor = torch.FloatTensor(np.array(hunter_obs)).to(self.device)
            self.hunter_net.eval()
            with torch.no_grad():
                actions, log_probs, values, _ = self.hunter_net(
                    hunter_obs_tensor, deterministic
                )
                actions = actions.cpu().numpy()
                log_probs = log_probs.cpu().numpy()
                values = values.cpu().numpy()

            for i, entity_id in enumerate(hunter_ids):
                actions_dict[entity_id] = actions[i]
                log_probs_dict[entity_id] = log_probs[i]
                values_dict[entity_id] = values[i]

        # 批量处理猎物
        if prey_obs:
            prey_obs_tensor = torch.FloatTensor(np.array(prey_obs)).to(self.device)
            self.prey_net.eval()
            with torch.no_grad():
                actions, log_probs, values, _ = self.prey_net(
                    prey_obs_tensor, deterministic
                )
                actions = actions.cpu().numpy()
                log_probs = log_probs.cpu().numpy()
                values = values.cpu().numpy()

            for i, entity_id in enumerate(prey_ids):
                actions_dict[entity_id] = actions[i]
                log_probs_dict[entity_id] = log_probs[i]
                values_dict[entity_id] = values[i]

        return actions_dict, log_probs_dict, values_dict

    def save(self, path: str):
        """保存模型"""
        torch.save(
            {
                "hunter_net": self.hunter_net.state_dict(),
                "prey_net": self.prey_net.state_dict(),
            },
            path,
        )
        print(f"模型已保存到 {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.hunter_net.load_state_dict(checkpoint["hunter_net"])
        self.prey_net.load_state_dict(checkpoint["prey_net"])
        print(f"模型已从 {path} 加载")
