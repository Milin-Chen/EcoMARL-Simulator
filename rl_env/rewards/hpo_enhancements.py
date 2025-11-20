"""
HPO启发的奖励增强模块
Hunter-Prey Optimization Inspired Reward Enhancements
"""

import numpy as np
import math
from typing import Dict, List, Tuple
from collections import deque


class AdaptiveRewardScaling:
    """自适应奖励缩放 (HPO能量机制启发)

    根据训练进度动态调整奖励权重:
    - 早期: 高探索权重，鼓励多样化行为
    - 后期: 高利用权重，快速收敛到最优解
    """

    def __init__(self, total_steps: int = 100000):
        """
        Args:
            total_steps: 总训练步数
        """
        self.total_steps = total_steps

    def get_reward_weights(self, current_step: int) -> Dict[str, float]:
        """
        根据训练进度返回奖励权重

        Args:
            current_step: 当前训练步数

        Returns:
            权重字典
        """
        progress = min(current_step / self.total_steps, 1.0)

        # 早期探索权重高，后期利用权重高
        exploration_factor = 1.0 - progress
        exploitation_factor = 0.5 + 0.5 * progress  # [0.5, 1.0]

        weights = {
            # 探索型奖励 (早期高)
            'movement': 2.0 * exploration_factor,
            'turn': 5.0 * exploration_factor,

            # 利用型奖励 (后期高)
            'direction': 15.0 * exploitation_factor,
            'approach': 15.0 * exploitation_factor,
            'capture': 200.0 * (0.8 + 0.2 * progress),  # [0.8, 1.0]
            'escape': 15.0 * exploitation_factor,

            # 惩罚项 (后期强)
            'stationary': -3.0 * exploitation_factor,
            'low_speed': 2.0 * exploitation_factor,
        }

        return weights


class AdversarialBalancer:
    """对抗性奖励平衡器 (HPO双向对抗启发)

    监控猎手/猎物成功率，动态调整奖励以保持生态平衡
    """

    def __init__(self, history_window: int = 100, target_balance: float = 0.5):
        """
        Args:
            history_window: 统计窗口大小
            target_balance: 目标平衡点 (0.5 = 50-50)
        """
        self.history_window = history_window
        self.target_balance = target_balance

        self.outcomes = deque(maxlen=history_window)  # ('capture' | 'escape')

        self.hunter_success_rate = 0.5
        self.prey_survival_rate = 0.5

    def update(self, outcome: str):
        """
        更新统计数据

        Args:
            outcome: 'capture' (猎手成功) 或 'escape' (猎物逃脱)
        """
        self.outcomes.append(outcome)

        if len(self.outcomes) >= 10:  # 至少10个样本才统计
            captures = sum(1 for o in self.outcomes if o == 'capture')
            total = len(self.outcomes)

            self.hunter_success_rate = captures / total
            self.prey_survival_rate = 1.0 - self.hunter_success_rate

    def get_balance_multipliers(self) -> Tuple[float, float]:
        """
        获取平衡系数

        Returns:
            (hunter_multiplier, prey_multiplier)
        """
        # 如果猎手太强，降低猎手奖励/增加猎物奖励
        # 如果猎物太强，增加猎手奖励/降低猎物奖励

        hunter_mult = 1.0 + (self.target_balance - self.hunter_success_rate) * 0.5
        prey_mult = 1.0 + (self.target_balance - self.prey_survival_rate) * 0.5

        # 限制范围 [0.7, 1.3]
        hunter_mult = np.clip(hunter_mult, 0.7, 1.3)
        prey_mult = np.clip(prey_mult, 0.7, 1.3)

        return hunter_mult, prey_mult

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'hunter_success_rate': self.hunter_success_rate,
            'prey_survival_rate': self.prey_survival_rate,
            'sample_count': len(self.outcomes),
        }


class DistanceProgressTracker:
    """距离进度追踪器 (HPO距离最小化启发)

    追踪智能体与目标的距离变化，奖励有效的接近/远离行为
    """

    def __init__(self, decay: float = 0.99):
        """
        Args:
            decay: 衰减因子 (模拟HPO能量降低)
        """
        self.decay = decay
        self.prev_distances = {}  # entity_id -> distance
        self.step_count = 0

    def compute_progress_reward(
        self,
        entity_id: str,
        entity_type: str,
        current_distance: float,
        scale: float = 10.0
    ) -> float:
        """
        计算距离进度奖励

        Args:
            entity_id: 实体ID
            entity_type: 'hunter' 或 'prey'
            current_distance: 当前距离
            scale: 奖励缩放系数

        Returns:
            进度奖励值
        """
        reward = 0.0

        prev_distance = self.prev_distances.get(entity_id)

        if prev_distance is not None:
            # 距离变化
            distance_delta = prev_distance - current_distance

            if entity_type == 'hunter':
                # 猎手: 奖励接近 (距离减小)
                if distance_delta > 0:
                    reward = scale * min(distance_delta / 10.0, 1.0)
                else:
                    # 轻微惩罚远离
                    reward = -scale * 0.3 * min(abs(distance_delta) / 10.0, 1.0)

            else:  # prey
                # 猎物: 奖励远离 (距离增大)
                if distance_delta < 0:
                    reward = scale * min(abs(distance_delta) / 10.0, 1.0)
                else:
                    # 轻微惩罚接近
                    reward = -scale * 0.3 * min(distance_delta / 10.0, 1.0)

        # 更新记录
        self.prev_distances[entity_id] = current_distance

        # 应用衰减 (模拟能量降低)
        decay_factor = self.decay ** (self.step_count / 100)
        reward *= decay_factor

        return reward

    def step(self):
        """每步调用"""
        self.step_count += 1

    def reset(self):
        """重置追踪器"""
        self.prev_distances.clear()
        self.step_count = 0


class HPORewardEnhancer:
    """HPO奖励增强器 - 统一接口

    整合所有HPO启发的增强组件
    """

    def __init__(
        self,
        total_steps: int = 100000,
        enable_adaptive: bool = True,
        enable_balancing: bool = True,
        enable_distance: bool = True,
    ):
        """
        Args:
            total_steps: 总训练步数
            enable_adaptive: 启用自适应权重
            enable_balancing: 启用对抗平衡
            enable_distance: 启用距离进度追踪
        """
        self.current_step = 0
        self.total_steps = total_steps

        # 组件开关
        self.enable_adaptive = enable_adaptive
        self.enable_balancing = enable_balancing
        self.enable_distance = enable_distance

        # 初始化组件
        if enable_adaptive:
            self.adaptive_scaler = AdaptiveRewardScaling(total_steps)

        if enable_balancing:
            self.adversarial_balancer = AdversarialBalancer()

        if enable_distance:
            self.distance_tracker = DistanceProgressTracker()

    def get_reward_weights(self) -> Dict[str, float]:
        """获取当前训练阶段的奖励权重"""
        if self.enable_adaptive:
            return self.adaptive_scaler.get_reward_weights(self.current_step)
        else:
            # 返回默认权重
            return {
                'movement': 2.0,
                'turn': 5.0,
                'direction': 15.0,
                'approach': 15.0,
                'capture': 200.0,
                'escape': 15.0,
                'stationary': -3.0,
                'low_speed': 2.0,
            }

    def get_balance_multipliers(self) -> Tuple[float, float]:
        """获取对抗平衡系数"""
        if self.enable_balancing:
            return self.adversarial_balancer.get_balance_multipliers()
        else:
            return 1.0, 1.0

    def compute_distance_progress_reward(
        self,
        entity_id: str,
        entity_type: str,
        current_distance: float,
        scale: float = 10.0
    ) -> float:
        """计算距离进度奖励"""
        if self.enable_distance:
            return self.distance_tracker.compute_progress_reward(
                entity_id, entity_type, current_distance, scale
            )
        else:
            return 0.0

    def update_outcome(self, outcome: str):
        """更新对抗结果统计"""
        if self.enable_balancing:
            self.adversarial_balancer.update(outcome)

    def step(self):
        """每步调用"""
        self.current_step += 1

        if self.enable_distance:
            self.distance_tracker.step()

    def reset(self):
        """每个episode重置"""
        if self.enable_distance:
            self.distance_tracker.reset()

    def get_stats(self) -> Dict[str, any]:
        """获取统计信息"""
        stats = {
            'current_step': self.current_step,
            'progress': min(self.current_step / self.total_steps, 1.0),
        }

        if self.enable_adaptive:
            stats['weights'] = self.get_reward_weights()

        if self.enable_balancing:
            stats['balance'] = self.adversarial_balancer.get_stats()

        return stats
