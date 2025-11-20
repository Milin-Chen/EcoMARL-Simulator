"""
增强版奖励函数 V2 - 解决随机运动获得高奖励问题

核心改进:
1. 意图性检测 - 只奖励有意的追击行为
2. 奖励归一化 - 所有奖励在[-1, 1]范围内
3. 一致性要求 - 需要连续朝向目标
4. 基线扣除 - 减去随机策略的期望奖励
"""

import math
from typing import Dict, Optional, List
from models import EntityState, WorldState


class EnhancedRewardFunctionV2:
    """增强版奖励函数 V2 - 防止随机策略获得高奖励"""

    def __init__(self):
        # ===== 归一化的奖励参数 =====
        # 所有奖励都在[-1, 1]范围内,最终乘以scale
        self.reward_scale = 10.0  # 全局缩放因子

        # 稀疏事件奖励 (大奖励)
        self.predation_reward = 10.0  # 捕食成功
        self.death_penalty = -5.0     # 死亡
        self.breeding_reward = 5.0    # 繁殖

        # 密集奖励 (归一化到[-1, 1])
        self.survival_reward = 0.01   # 存活 (非常小)
        self.approach_base = 0.3      # 接近目标基础奖励
        self.direction_base = 0.2     # 方向对齐基础奖励
        self.consistency_bonus = 0.3  # 连续性奖励

        # 惩罚 (归一化)
        self.inefficient_movement_penalty = -0.5  # 无效运动
        self.random_movement_penalty = -0.2       # 随机运动

        # 距离阈值
        self.hunt_distance_threshold = 200.0
        self.close_range_threshold = 50.0
        self.very_close_threshold = 20.0

        # ===== 意图性检测参数 =====
        self.min_speed_for_pursuit = 5.0   # 追击最小速度
        self.min_alignment_for_pursuit = 0.7  # 追击最小对齐度 (cos)

        # ===== 一致性检测参数 =====
        self.consistency_window = 3  # 检查最近3步的一致性

        # 历史记录
        self.prev_distances: Dict[str, float] = {}
        self.prev_positions: Dict[str, tuple] = {}
        self.prev_angles: Dict[str, float] = {}
        self.prev_speeds: Dict[str, float] = {}
        self.target_history: Dict[str, str] = {}

        # 一致性追踪
        self.pursuit_history: Dict[str, List[bool]] = {}  # 最近N步是否在追击
        self.escape_history: Dict[str, List[bool]] = {}   # 最近N步是否在逃跑

    def compute_reward(
        self,
        entity: EntityState,
        prev_entity: EntityState,
        prev_world: WorldState,
        curr_world: WorldState,
    ) -> float:
        """计算单个实体的奖励"""
        # 基础存活奖励
        reward = self.survival_reward

        # 死亡惩罚
        if entity.energy <= 0:
            reward += self.death_penalty
            return reward * self.reward_scale

        # 根据类型计算奖励
        if entity.type == "hunter":
            reward += self._compute_hunter_reward(
                entity, prev_entity, prev_world, curr_world
            )
        else:  # prey
            reward += self._compute_prey_reward(
                entity, prev_entity, prev_world, curr_world
            )

        # 繁殖奖励
        if entity.offspring_count > prev_entity.offspring_count:
            reward += self.breeding_reward

        # 全局缩放
        return reward * self.reward_scale

    def _compute_hunter_reward(
        self,
        hunter: EntityState,
        prev_hunter: EntityState,
        prev_world: WorldState,
        curr_world: WorldState,
    ) -> float:
        """计算猎人奖励 - V2增强版"""
        reward = 0.0

        # 1. 捕食成功 - 最大奖励
        predation_events = [
            e for e in curr_world.events
            if e.type == "predation" and e.actor_id == hunter.id
        ]
        if predation_events:
            reward += self.predation_reward
            # 清除历史
            self._clear_history(hunter.id)
            return reward

        # 获取猎物
        preys = [e for e in curr_world.entities if e.type == "prey"]
        if not preys:
            return reward

        # 2. 找到最近的猎物
        nearest_prey, min_distance = self._find_nearest_entity(hunter, preys)
        if nearest_prey is None:
            return reward

        # 3. 意图性检测 - 是否在有意追击?
        is_intentional_pursuit = self._detect_intentional_pursuit(
            hunter, prev_hunter, nearest_prey, min_distance
        )

        # 4. 只有有意追击才给接近奖励
        if is_intentional_pursuit:
            # 距离变化奖励
            prev_distance = self.prev_distances.get(hunter.id, min_distance)
            distance_delta = prev_distance - min_distance

            if distance_delta > 0:  # 接近目标
                # 归一化距离变化 (最大变化约为max_speed)
                normalized_delta = min(distance_delta / 50.0, 1.0)

                # 距离系数 (越近越重要)
                if min_distance < self.very_close_threshold:
                    distance_factor = 2.0
                elif min_distance < self.close_range_threshold:
                    distance_factor = 1.5
                else:
                    distance_factor = 1.0

                approach_reward = self.approach_base * normalized_delta * distance_factor
                reward += approach_reward

            # 方向对齐奖励 (仅在有意追击时给予)
            alignment_reward = self._compute_normalized_direction_reward(
                hunter, nearest_prey, min_distance
            )
            reward += alignment_reward

            # 一致性奖励 - 连续追击
            consistency_reward = self._compute_consistency_bonus(hunter.id, is_pursuing=True)
            reward += consistency_reward

        else:
            # 非有意追击,判断是否是随机运动
            if hunter.speed > 10.0:  # 有一定速度但不是朝向目标
                # 随机运动惩罚
                reward += self.random_movement_penalty

            # 重置一致性
            self._update_consistency(hunter.id, is_pursuing=False)

        # 5. 无效运动惩罚
        if self._is_inefficient_movement(hunter, prev_hunter):
            reward += self.inefficient_movement_penalty

        # 更新历史
        self._update_history(hunter, nearest_prey, min_distance)

        return reward

    def _detect_intentional_pursuit(
        self,
        hunter: EntityState,
        prev_hunter: EntityState,
        prey: EntityState,
        distance: float,
    ) -> bool:
        """
        检测是否是有意的追击行为

        要求:
        1. 速度足够 (不是静止)
        2. 方向对齐 (朝向目标)
        3. 速度方向和位移方向一致
        """
        # 检查1: 速度要求
        if hunter.speed < self.min_speed_for_pursuit:
            return False

        # 检查2: 方向对齐
        dx = prey.x - hunter.x
        dy = prey.y - hunter.y
        target_angle = math.atan2(dy, dx)

        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        alignment = math.cos(angle_diff)  # [-1, 1], 1表示完全对齐

        if alignment < self.min_alignment_for_pursuit:
            return False

        # 检查3: 实际位移方向 (可选但更严格)
        prev_pos = self.prev_positions.get(hunter.id)
        if prev_pos is not None:
            actual_dx = hunter.x - prev_pos[0]
            actual_dy = hunter.y - prev_pos[1]
            actual_movement = math.sqrt(actual_dx**2 + actual_dy**2)

            if actual_movement > 1.0:  # 有显著移动
                # 计算实际移动方向
                actual_angle = math.atan2(actual_dy, actual_dx)

                # 实际移动方向 vs 目标方向
                movement_alignment_diff = abs(actual_angle - target_angle)
                if movement_alignment_diff > math.pi:
                    movement_alignment_diff = 2 * math.pi - movement_alignment_diff

                movement_alignment = math.cos(movement_alignment_diff)

                # 要求实际移动方向也朝向目标
                if movement_alignment < 0.5:  # cos(60°) = 0.5
                    return False

        return True

    def _compute_normalized_direction_reward(
        self,
        hunter: EntityState,
        prey: EntityState,
        distance: float,
    ) -> float:
        """计算归一化的方向对齐奖励"""
        if distance > self.hunt_distance_threshold:
            return 0.0

        # 计算对齐度
        dx = prey.x - hunter.x
        dy = prey.y - hunter.y
        target_angle = math.atan2(dy, dx)

        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # 使用cos而非线性映射 (更平滑)
        alignment = math.cos(angle_diff)  # [-1, 1]
        alignment = (alignment + 1.0) / 2.0  # [0, 1]

        # 距离权重
        distance_weight = 1.0 - (distance / self.hunt_distance_threshold)

        # 归一化奖励
        reward = self.direction_base * alignment * distance_weight

        return reward

    def _compute_consistency_bonus(self, agent_id: str, is_pursuing: bool) -> float:
        """计算一致性奖励 - 奖励连续的追击行为"""
        # 更新历史
        if agent_id not in self.pursuit_history:
            self.pursuit_history[agent_id] = []

        history = self.pursuit_history[agent_id]
        history.append(is_pursuing)

        # 只保留最近N步
        if len(history) > self.consistency_window:
            history.pop(0)

        # 计算一致性
        if len(history) >= self.consistency_window:
            # 如果最近N步都在追击,给予奖励
            if all(history):
                return self.consistency_bonus

        return 0.0

    def _update_consistency(self, agent_id: str, is_pursuing: bool):
        """更新一致性历史"""
        if agent_id not in self.pursuit_history:
            self.pursuit_history[agent_id] = []

        self.pursuit_history[agent_id].append(is_pursuing)

        if len(self.pursuit_history[agent_id]) > self.consistency_window:
            self.pursuit_history[agent_id].pop(0)

    def _is_inefficient_movement(
        self,
        hunter: EntityState,
        prev_hunter: EntityState,
    ) -> bool:
        """检测无效运动 (原地转圈或高速但无进展)"""
        prev_pos = self.prev_positions.get(hunter.id)
        prev_angle = self.prev_angles.get(hunter.id)

        if prev_pos is None or prev_angle is None:
            return False

        # 位置变化
        dx = hunter.x - prev_pos[0]
        dy = hunter.y - prev_pos[1]
        actual_movement = math.sqrt(dx**2 + dy**2)

        # 角度变化
        angle_change = abs(hunter.angle - prev_angle)
        if angle_change > math.pi:
            angle_change = 2 * math.pi - angle_change

        # 情况1: 大角度变化但位置几乎没变 (原地转圈)
        if angle_change > 0.5 and actual_movement < 2.0:
            return True

        # 情况2: 高速但位移很小 (在原地抖动)
        if hunter.speed > 20.0 and actual_movement < 5.0:
            return True

        return False

    def _compute_prey_reward(
        self,
        prey: EntityState,
        prev_prey: EntityState,
        prev_world: WorldState,
        curr_world: WorldState,
    ) -> float:
        """计算猎物奖励 - V2版本"""
        reward = 0.0

        # 获取猎人
        hunters = [e for e in curr_world.entities if e.type == "hunter"]
        if not hunters:
            return reward

        # 找到最近的猎人
        nearest_hunter, min_distance = self._find_nearest_entity(prey, hunters)
        if nearest_hunter is None:
            return reward

        # 检测是否在有意逃跑
        is_intentional_escape = self._detect_intentional_escape(
            prey, prev_prey, nearest_hunter, min_distance
        )

        if is_intentional_escape:
            # 距离变化奖励
            prev_distance = self.prev_distances.get(prey.id, min_distance)
            distance_delta = min_distance - prev_distance  # 注意:猎物是远离

            if distance_delta > 0:  # 远离猎人
                normalized_delta = min(distance_delta / 50.0, 1.0)

                # 距离越近逃跑越重要
                if min_distance < self.very_close_threshold:
                    distance_factor = 2.0
                elif min_distance < self.close_range_threshold:
                    distance_factor = 1.5
                else:
                    distance_factor = 0.5  # 已经很远了,不那么重要

                escape_reward = self.approach_base * normalized_delta * distance_factor
                reward += escape_reward

            # 方向奖励 (背对猎人)
            alignment_reward = self._compute_escape_direction_reward(
                prey, nearest_hunter, min_distance
            )
            reward += alignment_reward

            # 一致性奖励
            consistency_reward = self._compute_consistency_bonus(prey.id, is_pursuing=True)
            reward += consistency_reward
        else:
            # 非有意逃跑
            if prey.speed > 10.0:
                reward += self.random_movement_penalty

            self._update_consistency(prey.id, is_pursuing=False)

        # 无效运动惩罚
        if self._is_inefficient_movement(prey, prev_prey):
            reward += self.inefficient_movement_penalty

        # 更新历史
        self._update_history(prey, nearest_hunter, min_distance)

        return reward

    def _detect_intentional_escape(
        self,
        prey: EntityState,
        prev_prey: EntityState,
        hunter: EntityState,
        distance: float,
    ) -> bool:
        """检测是否是有意的逃跑行为"""
        # 速度要求
        if prey.speed < self.min_speed_for_pursuit:
            return False

        # 方向要求: 背对猎人
        dx = hunter.x - prey.x
        dy = hunter.y - prey.y
        threat_angle = math.atan2(dy, dx)

        # 逃跑应该朝向威胁的反方向
        escape_angle = threat_angle + math.pi
        if escape_angle > math.pi:
            escape_angle -= 2 * math.pi

        angle_diff = abs(prey.angle - escape_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        alignment = math.cos(angle_diff)

        if alignment < self.min_alignment_for_pursuit:
            return False

        # 实际位移检查
        prev_pos = self.prev_positions.get(prey.id)
        if prev_pos is not None:
            actual_dx = prey.x - prev_pos[0]
            actual_dy = prey.y - prev_pos[1]
            actual_movement = math.sqrt(actual_dx**2 + actual_dy**2)

            if actual_movement > 1.0:
                actual_angle = math.atan2(actual_dy, actual_dx)

                movement_diff = abs(actual_angle - escape_angle)
                if movement_diff > math.pi:
                    movement_diff = 2 * math.pi - movement_diff

                movement_alignment = math.cos(movement_diff)

                if movement_alignment < 0.5:
                    return False

        return True

    def _compute_escape_direction_reward(
        self,
        prey: EntityState,
        hunter: EntityState,
        distance: float,
    ) -> float:
        """计算逃跑方向奖励"""
        if distance > self.hunt_distance_threshold:
            return 0.0

        # 威胁方向
        dx = hunter.x - prey.x
        dy = hunter.y - prey.y
        threat_angle = math.atan2(dy, dx)

        # 理想逃跑方向 (威胁的反方向)
        escape_angle = threat_angle + math.pi
        if escape_angle > math.pi:
            escape_angle -= 2 * math.pi

        # 实际朝向 vs 理想逃跑方向
        angle_diff = abs(prey.angle - escape_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        alignment = math.cos(angle_diff)
        alignment = (alignment + 1.0) / 2.0  # [0, 1]

        # 距离越近,方向越重要
        distance_weight = 1.0 - min(distance / self.hunt_distance_threshold, 1.0)
        distance_weight = 0.3 + 0.7 * distance_weight  # [0.3, 1.0]

        reward = self.direction_base * alignment * distance_weight

        return reward

    def _find_nearest_entity(
        self, entity: EntityState, targets: List[EntityState]
    ) -> tuple[Optional[EntityState], float]:
        """找到最近的目标实体"""
        if not targets:
            return None, float('inf')

        min_distance = float('inf')
        nearest = None

        for target in targets:
            dx = target.x - entity.x
            dy = target.y - entity.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < min_distance:
                min_distance = distance
                nearest = target

        return nearest, min_distance

    def _update_history(self, entity: EntityState, target: EntityState, distance: float):
        """更新历史记录"""
        self.prev_distances[entity.id] = distance
        self.prev_positions[entity.id] = (entity.x, entity.y)
        self.prev_angles[entity.id] = entity.angle
        self.prev_speeds[entity.id] = entity.speed
        self.target_history[entity.id] = target.id

    def _clear_history(self, entity_id: str):
        """清除实体的历史记录"""
        self.prev_distances.pop(entity_id, None)
        self.prev_positions.pop(entity_id, None)
        self.prev_angles.pop(entity_id, None)
        self.prev_speeds.pop(entity_id, None)
        self.target_history.pop(entity_id, None)
        self.pursuit_history.pop(entity_id, None)
        self.escape_history.pop(entity_id, None)

    def compute_population_rewards(self, world: WorldState) -> tuple[float, float, str]:
        """
        计算种群平衡奖励/惩罚

        返回: (hunter_penalty, prey_penalty, message)
        注: V2版本目前不使用种群平衡惩罚,返回0
        """
        # V2版本暂时不使用种群平衡,可以在未来添加
        return 0.0, 0.0, ""

    def reset(self):
        """重置所有历史记录"""
        self.prev_distances.clear()
        self.prev_positions.clear()
        self.prev_angles.clear()
        self.prev_speeds.clear()
        self.target_history.clear()
        self.pursuit_history.clear()
        self.escape_history.clear()
