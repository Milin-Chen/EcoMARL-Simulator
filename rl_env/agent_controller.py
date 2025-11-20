"""智能体控制器 - 将RL模型接入模拟器"""

import numpy as np
from typing import Dict, List
try:
    from .training import SharedPolicy
except ImportError:
    SharedPolicy = None
from .observations import ObservationSpace
from models import EntityState


class RLAgentController:
    """
    强化学习智能体控制器
    用于将训练好的模型接入到模拟器的运动系统
    """

    def __init__(
        self,
        policy: SharedPolicy,
        obs_space: ObservationSpace,
        deterministic: bool = False,
    ):
        """
        Args:
            policy: 共享策略网络
            obs_space: 观察空间
            deterministic: 是否使用确定性策略（训练时False，评估时True）
        """
        self.policy = policy
        self.obs_space = obs_space
        self.deterministic = deterministic

    def get_actions(
        self, entities: List[EntityState], all_entities: List[EntityState]
    ) -> Dict[str, np.ndarray]:
        """
        为所有实体生成动作

        Args:
            entities: 需要控制的实体列表
            all_entities: 所有实体列表（用于计算观察）

        Returns:
            actions_dict: {entity_id: action} 字典
                action = [speed_delta, angular_velocity_delta]
        """
        if not entities:
            return {}

        # 准备观察数据
        obs_dict = {}
        for entity in entities:
            obs = self.obs_space.get_observation(entity, all_entities)
            obs_dict[entity.id] = (obs, entity.type)

        # 批量获取动作
        actions_dict, _, _ = self.policy.get_actions_batch(
            obs_dict, self.deterministic
        )

        return actions_dict

    def apply_actions(
        self, entities: List[EntityState], all_entities: List[EntityState]
    ):
        """
        为实体生成并应用动作

        Args:
            entities: 需要控制的实体列表
            all_entities: 所有实体列表
        """
        actions = self.get_actions(entities, all_entities)

        for entity in entities:
            if entity.id in actions:
                action = actions[entity.id]
                self._apply_single_action(entity, action)

    def _apply_single_action(self, entity: EntityState, action: np.ndarray):
        """
        应用单个动作到实体

        Args:
            entity: 实体
            action: 动作 [speed_delta, angular_velocity_delta]
        """
        from config import AgentConfig

        cfg = AgentConfig()

        # 应用速度变化
        speed_delta = float(action[0])
        entity.speed = np.clip(
            entity.speed + speed_delta,
            0.0,
            (
                cfg.HUNTER_SPEED_MAX
                if entity.type == "hunter"
                else cfg.PREY_SPEED_MAX
            ),
        )

        # 应用角速度变化
        angular_velocity_delta = float(action[1])
        entity.angular_velocity = np.clip(
            entity.angular_velocity + angular_velocity_delta,
            -cfg.HUNTER_ANGULAR_VELOCITY_MAX,
            cfg.HUNTER_ANGULAR_VELOCITY_MAX,
        )

    def set_deterministic(self, deterministic: bool):
        """设置是否使用确定性策略"""
        self.deterministic = deterministic


class HybridController:
    """
    混合控制器 - 支持部分实体使用RL，部分使用规则
    """

    def __init__(
        self,
        rl_controller: RLAgentController = None,
        use_rl_for_hunters: bool = True,
        use_rl_for_prey: bool = True,
    ):
        """
        Args:
            rl_controller: RL控制器
            use_rl_for_hunters: 猎人是否使用RL
            use_rl_for_prey: 猎物是否使用RL
        """
        self.rl_controller = rl_controller
        self.use_rl_for_hunters = use_rl_for_hunters
        self.use_rl_for_prey = use_rl_for_prey

    def apply_actions(
        self, entities: List[EntityState], all_entities: List[EntityState]
    ):
        """应用动作到所有实体"""
        # 分类实体
        rl_entities = []
        rule_entities = []

        for entity in entities:
            if entity.type == "hunter" and self.use_rl_for_hunters:
                rl_entities.append(entity)
            elif entity.type == "prey" and self.use_rl_for_prey:
                rl_entities.append(entity)
            else:
                rule_entities.append(entity)

        # RL控制
        if rl_entities and self.rl_controller:
            self.rl_controller.apply_actions(rl_entities, all_entities)

        # 规则控制
        if rule_entities:
            self._apply_rule_based_actions(rule_entities, all_entities)

    def _apply_rule_based_actions(
        self, entities: List[EntityState], all_entities: List[EntityState]
    ):
        """
        基于规则的控制（简单的追逐/逃避行为）

        这是一个简单的示例，可以根据需要自定义
        """
        for entity in entities:
            if entity.type == "hunter":
                # 猎人：追逐最近的猎物
                self._chase_nearest(entity, all_entities, "prey")
            else:
                # 猎物：逃离最近的猎人
                self._flee_from_nearest(entity, all_entities, "hunter")

    def _chase_nearest(
        self, entity: EntityState, all_entities: List[EntityState], target_type: str
    ):
        """追逐最近的目标"""
        targets = [e for e in all_entities if e.type == target_type]
        if not targets:
            # 随机游走
            entity.angular_velocity = np.random.uniform(-0.1, 0.1)
            entity.speed = 30.0
            return

        # 找到最近的目标
        nearest = min(
            targets,
            key=lambda t: (t.x - entity.x) ** 2 + (t.y - entity.y) ** 2,
        )

        # 计算朝向目标的角度
        dx = nearest.x - entity.x
        dy = nearest.y - entity.y
        target_angle = np.arctan2(dy, dx)

        # 调整角速度朝向目标
        angle_diff = self._normalize_angle(target_angle - entity.angle)
        entity.angular_velocity = np.clip(angle_diff * 0.5, -0.3, 0.3)

        # 加速
        entity.speed = min(entity.speed + 5.0, 50.0)

    def _flee_from_nearest(
        self, entity: EntityState, all_entities: List[EntityState], threat_type: str
    ):
        """逃离最近的威胁"""
        threats = [e for e in all_entities if e.type == threat_type]
        if not threats:
            # 随机游走
            entity.angular_velocity = np.random.uniform(-0.1, 0.1)
            entity.speed = 20.0
            return

        # 找到最近的威胁
        nearest = min(
            threats,
            key=lambda t: (t.x - entity.x) ** 2 + (t.y - entity.y) ** 2,
        )

        # 计算远离威胁的角度
        dx = entity.x - nearest.x
        dy = entity.y - nearest.y
        escape_angle = np.arctan2(dy, dx)

        # 调整角速度远离威胁
        angle_diff = self._normalize_angle(escape_angle - entity.angle)
        entity.angular_velocity = np.clip(angle_diff * 0.5, -0.3, 0.3)

        # 加速
        entity.speed = min(entity.speed + 5.0, 35.0)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """归一化角度到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
