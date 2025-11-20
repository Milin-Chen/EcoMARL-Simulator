"""PPO训练器 - 支持多智能体参数共享"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from .networks import SharedPolicy


class RolloutBuffer:
    """经验回放缓冲区"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置缓冲区"""
        # 按物种分别存储
        self.hunter_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }
        self.prey_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
        }

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        entity_type: str,
    ):
        """添加经验"""
        data = self.hunter_data if entity_type == "hunter" else self.prey_data

        data["observations"].append(obs)
        data["actions"].append(action)
        data["rewards"].append(reward)
        data["values"].append(value)
        data["log_probs"].append(log_prob)
        data["dones"].append(done)

    def get_data(self, entity_type: str) -> Dict[str, np.ndarray]:
        """获取数据（转换为numpy数组）"""
        data = self.hunter_data if entity_type == "hunter" else self.prey_data

        if len(data["observations"]) == 0:
            return None

        return {
            "observations": np.array(data["observations"], dtype=np.float32),
            "actions": np.array(data["actions"], dtype=np.float32),
            "rewards": np.array(data["rewards"], dtype=np.float32),
            "values": np.array(data["values"], dtype=np.float32),
            "log_probs": np.array(data["log_probs"], dtype=np.float32),
            "dones": np.array(data["dones"], dtype=np.float32),
        }

    def compute_returns_and_advantages(
        self, last_values: Dict[str, float], gamma: float = 0.99, gae_lambda: float = 0.95
    ):
        """
        计算回报和优势函数（使用GAE）

        Args:
            last_values: 最后状态的价值 {"hunter": value, "prey": value}
            gamma: 折扣因子
            gae_lambda: GAE参数
        """
        for entity_type in ["hunter", "prey"]:
            data = self.get_data(entity_type)
            if data is None:
                continue

            rewards = data["rewards"]
            values = data["values"]
            dones = data["dones"]

            # 添加最后的价值估计
            last_value = last_values.get(entity_type, 0.0)
            values_with_last = np.append(values, last_value)

            # 计算GAE
            advantages = np.zeros_like(rewards)
            last_gae_lam = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = values_with_last[t + 1]
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_values = values[t + 1]

                delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = (
                    delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
                )

            # 计算回报
            returns = advantages + values

            # 存储回计算的数据
            if entity_type == "hunter":
                self.hunter_data["advantages"] = advantages
                self.hunter_data["returns"] = returns
            else:
                self.prey_data["advantages"] = advantages
                self.prey_data["returns"] = returns


class PPOTrainer:
    """PPO训练器 - 支持参数共享的多智能体训练"""

    def __init__(
        self,
        policy: SharedPolicy,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # 为两个网络创建独立的优化器
        self.hunter_optimizer = optim.Adam(
            policy.hunter_net.parameters(), lr=learning_rate
        )
        self.prey_optimizer = optim.Adam(
            policy.prey_net.parameters(), lr=learning_rate
        )

        # 统计信息
        self.train_stats = defaultdict(list)

    def train_step(self, buffer: RolloutBuffer, last_values: Dict[str, float]):
        """
        执行一次PPO更新

        Args:
            buffer: 经验缓冲区
            last_values: 最后状态的价值估计
        """
        # 计算回报和优势
        buffer.compute_returns_and_advantages(last_values, self.gamma, self.gae_lambda)

        # 分别训练两个网络
        hunter_stats = self._update_network(buffer, "hunter", self.hunter_optimizer)
        prey_stats = self._update_network(buffer, "prey", self.prey_optimizer)

        # 记录统计信息
        for key, value in hunter_stats.items():
            self.train_stats[f"hunter_{key}"].append(value)
        for key, value in prey_stats.items():
            self.train_stats[f"prey_{key}"].append(value)

        return {**hunter_stats, **prey_stats}

    def _update_network(
        self, buffer: RolloutBuffer, entity_type: str, optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """更新单个网络"""
        data = buffer.get_data(entity_type)
        if data is None:
            return {}

        # 检查数据量
        if len(data["observations"]) < 2:
            return {}

        # 获取数据
        obs = torch.FloatTensor(data["observations"]).to(self.policy.device)
        actions = torch.FloatTensor(data["actions"]).to(self.policy.device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(self.policy.device)

        # 检查输入数据
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"警告: {entity_type} 观察数据包含NaN/Inf，跳过此次更新")
            return {}
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print(f"警告: {entity_type} 动作数据包含NaN/Inf，跳过此次更新")
            return {}

        advantages = data["advantages"] if "advantages" in data else data["rewards"]
        returns = data["returns"] if "returns" in data else data["rewards"]

        # 标准化优势（更安全的版本）
        advantages = torch.FloatTensor(advantages).to(self.policy.device)
        adv_mean = advantages.mean()
        adv_std = advantages.std()

        # 如果标准差太小，不进行标准化
        if adv_std > 1e-6:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            # 标准差太小，只进行中心化
            advantages = advantages - adv_mean

        returns = torch.FloatTensor(returns).to(self.policy.device)

        # 获取网络
        network = self.policy.get_network(entity_type)

        # 多轮更新
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(obs))

            # 分批训练
            for start in range(0, len(obs), self.batch_size):
                end = min(start + self.batch_size, len(obs))
                batch_indices = indices[start:end]

                # 获取批次数据
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 评估当前策略
                log_probs, values, entropy = network.evaluate_actions(
                    batch_obs, batch_actions
                )

                # 计算比率
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # PPO裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # 熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # NaN检测
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: {entity_type} 网络检测到NaN/Inf，跳过此次更新")
                    print(f"  policy_loss: {policy_loss.item()}")
                    print(f"  value_loss: {value_loss.item()}")
                    print(f"  entropy_loss: {entropy_loss.item()}")
                    continue

                # 更新网络
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪并检查
                grad_norm = nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)

                # 检查梯度是否异常
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"警告: {entity_type} 网络检测到异常梯度，跳过此次更新")
                    optimizer.zero_grad()
                    continue

                optimizer.step()

                # 记录损失
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                n_updates += 1

        # 返回平均损失
        if n_updates > 0:
            return {
                f"{entity_type}_policy_loss": total_policy_loss / n_updates,
                f"{entity_type}_value_loss": total_value_loss / n_updates,
                f"{entity_type}_entropy_loss": total_entropy_loss / n_updates,
            }
        else:
            return {}

    def get_stats(self) -> Dict[str, float]:
        """获取最近的训练统计"""
        stats = {}
        for key, values in self.train_stats.items():
            if values:
                stats[key] = np.mean(values[-10:])  # 最近10次的平均值
        return stats

    def clear_stats(self):
        """清空统计信息"""
        self.train_stats.clear()
