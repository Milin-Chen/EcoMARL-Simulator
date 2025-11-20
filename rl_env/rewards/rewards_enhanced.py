"""
增强版奖励函数 - 修复训练问题

主要改进:
1. 增强向目标移动的奖励
2. 惩罚原地转圈和无效运动
3. 更明显的追击行为激励
4. 能量效率惩罚
5. 方向性奖励
"""

from __future__ import annotations
import numpy as np
import math
from typing import Dict, Optional, Tuple, List
from models import EntityState, WorldState


class EnhancedRewardFunction:
    """增强版奖励函数 - 解决训练问题"""

    def __init__(self):
        # ========== 基础奖励权重 ==========
        self.survival_reward = 0.1  # 降低存活奖励,避免passive策略
        self.death_penalty = -50.0  # 提高死亡惩罚
        self.breeding_reward = 20.0

        # ========== 猎人专属奖励 ==========
        self.predation_reward = 100.0  # 大幅提高捕食奖励
        self.chase_reward_scale = 5.0  # 提高追击奖励
        self.approach_reward = 2.0  # 新增: 向目标移动的基础奖励
        self.target_lock_reward = 1.0
        self.direction_alignment_scale = 3.0  # 新增: 方向对齐奖励

        # 能量惩罚
        self.useless_movement_penalty = -1.0  # 新增: 无效运动惩罚
        self.rotation_without_progress_penalty = -2.0  # 新增: 原地转圈惩罚

        # 距离阈值
        self.hunt_distance_threshold = 200.0
        self.close_range_threshold = 50.0  # 近距离阈值

        # ========== 猎物专属奖励 ==========
        self.escape_reward_scale = 3.0
        self.feeding_reward = 3.0
        self.danger_threshold = 120.0
        self.safe_distance_reward = 0.5

        # 猎物惩罚
        self.idle_penalty = -0.5  # 新增: 猎物idle惩罚

        # ========== 速度控制奖励 ==========
        self.speed_efficiency_bonus = 0.3

        # ========== 种群平衡惩罚 ==========
        self.extinction_penalty = -200.0
        self.low_population_penalty = -10.0

        # ========== 内部状态 ==========
        self.target_history: Dict[str, str] = {}
        self.prev_distances: Dict[str, float] = {}
        self.prev_positions: Dict[str, Tuple[float, float]] = {}  # 新增
        self.prev_angles: Dict[str, float] = {}  # 新增

    def compute_reward(
        self,
        entity: EntityState,
        prev_world: WorldState,
        curr_world: WorldState,
    ) -> float:
        """计算实体的奖励"""
        prev_entity = next(
            (e for e in prev_world.entities if e.id == entity.id), None
        )

        if prev_entity is None:
            return self.breeding_reward

        # 基础奖励
        reward = self.survival_reward

        # 死亡惩罚
        if entity.energy <= 0:
            reward += self.death_penalty
            return reward

        # 根据类型计算专属奖励
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

        return reward

    def _compute_hunter_reward(
        self,
        hunter: EntityState,
        prev_hunter: EntityState,
        prev_world: WorldState,
        curr_world: WorldState,
    ) -> float:
        """计算猎人奖励 - 增强版"""
        reward = 0.0

        # 1. 捕食成功 - 巨大奖励
        predation_events = [
            e
            for e in curr_world.events
            if e.type == "predation" and e.actor_id == hunter.id
        ]
        if predation_events:
            reward += self.predation_reward
            print(f"猎人 {hunter.id} 成功捕食! 奖励: {self.predation_reward}")
            self.target_history.pop(hunter.id, None)
            self.prev_distances.pop(hunter.id, None)
            self.prev_positions.pop(hunter.id, None)
            return reward

        # 获取猎物
        preys = [e for e in curr_world.entities if e.type == "prey"]
        if not preys:
            reward += self.low_population_penalty * 0.1
            return reward

        # 2. 找到最近的猎物
        nearest_prey, min_distance = self._find_nearest_entity(hunter, preys)
        if nearest_prey is None:
            return reward

        # 3. 向目标移动奖励 (核心改进!)
        prev_distance = self.prev_distances.get(hunter.id, min_distance)
        distance_delta = prev_distance - min_distance

        # 如果接近目标,给予奖励
        if distance_delta > 0:
            # 基础接近奖励
            base_approach = distance_delta * self.approach_reward

            # 距离越近,奖励越高
            proximity_factor = 1.0 + (1.0 - min_distance / self.hunt_distance_threshold)

            # 渐进式奖励
            if min_distance < self.hunt_distance_threshold:
                approach_reward = base_approach * proximity_factor * self.chase_reward_scale
                reward += approach_reward

                # 近距离额外奖励
                if min_distance < self.close_range_threshold:
                    bonus = (1.0 - min_distance / self.close_range_threshold) * 5.0
                    reward += bonus

        # 如果远离目标,给予惩罚
        elif distance_delta < -1.0:  # 远离超过1个单位
            reward += distance_delta * 0.5  # 轻微惩罚

        # 4. 方向对齐奖励 (新增!)
        direction_reward = self._compute_direction_alignment(
            hunter, nearest_prey, min_distance
        )
        reward += direction_reward

        # 5. 目标锁定奖励
        prev_target = self.target_history.get(hunter.id)
        if prev_target == nearest_prey.id:
            reward += self.target_lock_reward
        else:
            self.target_history[hunter.id] = nearest_prey.id

        # 6. 惩罚无效运动 (新增!)
        movement_penalty = self._compute_movement_efficiency_penalty(
            hunter, prev_hunter, distance_delta
        )
        reward += movement_penalty

        # 7. 视野锁定奖励
        if hasattr(hunter, 'rays') and hunter.rays:
            prey_detected = any(
                r.hit_id == nearest_prey.id for r in hunter.rays if r.hit_id
            )
            if prey_detected:
                reward += 1.0  # 提高视野锁定奖励

        # 更新历史
        self.prev_distances[hunter.id] = min_distance
        self.prev_positions[hunter.id] = (hunter.x, hunter.y)
        self.prev_angles[hunter.id] = hunter.angle

        return reward

    def _compute_direction_alignment(
        self,
        hunter: EntityState,
        target_prey: EntityState,
        distance: float,
    ) -> float:
        """计算方向对齐奖励 - 猎人朝向猎物方向"""
        if distance > self.hunt_distance_threshold:
            return 0.0

        # 计算目标方向
        dx = target_prey.x - hunter.x
        dy = target_prey.y - hunter.y
        target_angle = math.atan2(dy, dx)

        # 计算角度差
        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # 归一化到[0, 1], 0表示完全对齐
        alignment = 1.0 - (angle_diff / math.pi)

        # 距离越近,方向对齐越重要
        distance_factor = 1.0 - (distance / self.hunt_distance_threshold)

        # 方向对齐奖励
        direction_reward = alignment * distance_factor * self.direction_alignment_scale

        return direction_reward

    def _compute_movement_efficiency_penalty(
        self,
        hunter: EntityState,
        prev_hunter: EntityState,
        distance_delta: float,
    ) -> float:
        """计算运动效率惩罚 - 惩罚原地转圈和无效运动"""
        penalty = 0.0

        # 获取位置变化
        prev_pos = self.prev_positions.get(hunter.id)
        if prev_pos is None:
            return 0.0

        dx = hunter.x - prev_pos[0]
        dy = hunter.y - prev_pos[1]
        actual_movement = math.sqrt(dx * dx + dy * dy)

        # 获取角度变化
        prev_angle = self.prev_angles.get(hunter.id, hunter.angle)
        angle_change = abs(hunter.angle - prev_angle)
        if angle_change > math.pi:
            angle_change = 2 * math.pi - angle_change

        # 情况1: 原地转圈 (角度变化大但位置没变)
        if angle_change > 0.3 and actual_movement < 1.0:
            # 原地转圈,大量消耗能量但没有进展
            penalty += self.rotation_without_progress_penalty

        # 情况2: 高速运动但没有接近目标
        if hunter.speed > 20.0 and distance_delta < 0:
            # 快速移动但远离目标
            penalty += self.useless_movement_penalty * (hunter.speed / 50.0)

        # 情况3: 几乎不动 (idle)
        if actual_movement < 0.5 and hunter.speed < 5.0:
            penalty += -0.2  # 轻微惩罚idle

        return penalty

    def _compute_prey_reward(
        self,
        prey: EntityState,
        prev_prey: EntityState,
        prev_world: WorldState,
        curr_world: WorldState,
    ) -> float:
        """计算猎物奖励 - 增强版"""
        reward = 0.0

        # 获取猎人
        hunters = [e for e in curr_world.entities if e.type == "hunter"]
        if not hunters:
            # 无猎人时,鼓励觅食
            energy_gain = prey.energy - prev_prey.energy
            if energy_gain > 0:
                reward += energy_gain * 0.3
            return reward

        # 找到最近的猎人
        nearest_hunter, min_distance = self._find_nearest_entity(prey, hunters)
        if nearest_hunter is None:
            return reward

        # 逃离奖励
        prev_distance = self.prev_distances.get(prey.id, min_distance)
        distance_delta = min_distance - prev_distance

        # 在危险区域
        if min_distance < self.danger_threshold:
            danger_factor = 1.0 - (min_distance / self.danger_threshold)

            # 成功逃离
            if distance_delta > 0:
                escape_reward = distance_delta * self.escape_reward_scale * (1.0 + danger_factor * 3.0)
                reward += escape_reward

                # 极度危险时逃离额外奖励
                if min_distance < 50.0:
                    bonus = (1.0 - min_distance / 50.0) * 10.0
                    reward += bonus
            # 被接近
            else:
                reward += distance_delta * 1.0  # 惩罚被接近

        # 安全区域
        else:
            # 安全距离奖励
            if min_distance > self.danger_threshold * 1.5:
                reward += self.safe_distance_reward

            # 觅食奖励
            energy_gain = prey.energy - prev_prey.energy
            if energy_gain > 0.5:
                safety_factor = min(min_distance / self.danger_threshold, 2.0)
                reward += energy_gain * 0.2 * safety_factor

        # 方向奖励 - 背向猎人
        direction_reward = self._compute_escape_direction_reward(
            prey, nearest_hunter, min_distance
        )
        reward += direction_reward

        # 惩罚idle (在危险区域不动)
        if min_distance < self.danger_threshold:
            prev_pos = self.prev_positions.get(prey.id)
            if prev_pos:
                dx = prey.x - prev_pos[0]
                dy = prey.y - prev_pos[1]
                movement = math.sqrt(dx * dx + dy * dy)
                if movement < 1.0:
                    reward += self.idle_penalty * (1.0 - min_distance / self.danger_threshold)

        # 更新历史
        self.prev_distances[prey.id] = min_distance
        self.prev_positions[prey.id] = (prey.x, prey.y)
        self.prev_angles[prey.id] = prey.angle

        return reward

    def _compute_escape_direction_reward(
        self,
        prey: EntityState,
        hunter: EntityState,
        distance: float,
    ) -> float:
        """计算逃离方向奖励 - 猎物背向猎人"""
        if distance > self.danger_threshold:
            return 0.0

        # 计算逃离方向(与猎人相反)
        dx = prey.x - hunter.x
        dy = prey.y - hunter.y
        escape_angle = math.atan2(dy, dx)

        # 计算角度差
        angle_diff = abs(prey.angle - escape_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # 归一化,0表示背向猎人
        alignment = 1.0 - (angle_diff / math.pi)

        # 越危险,方向越重要
        danger_factor = 1.0 - (distance / self.danger_threshold)

        # 方向奖励
        direction_reward = alignment * danger_factor * 2.0

        return direction_reward

    def _find_nearest_entity(
        self, entity: EntityState, targets: List[EntityState]
    ) -> Tuple[Optional[EntityState], float]:
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

    def compute_population_rewards(
        self, world: WorldState
    ) -> Tuple[float, float, bool]:
        """计算种群平衡奖励/惩罚"""
        hunters = [e for e in world.entities if e.type == "hunter"]
        preys = [e for e in world.entities if e.type == "prey"]

        n_hunters = len(hunters)
        n_preys = len(preys)

        hunter_penalty = 0.0
        prey_penalty = 0.0
        should_terminate = False

        if n_hunters == 0:
            prey_penalty = self.extinction_penalty
            should_terminate = True
        elif n_preys == 0:
            hunter_penalty = self.extinction_penalty
            should_terminate = True

        if not should_terminate:
            if n_hunters < 3:
                hunter_penalty += self.low_population_penalty
            if n_preys < 6:
                prey_penalty += self.low_population_penalty

        return hunter_penalty, prey_penalty, should_terminate

    def reset(self):
        """重置内部状态"""
        self.target_history.clear()
        self.prev_distances.clear()
        self.prev_positions.clear()
        self.prev_angles.clear()

    def set_weights(self, **kwargs):
        """设置奖励权重"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
