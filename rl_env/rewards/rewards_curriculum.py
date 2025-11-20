"""
分阶段训练的奖励函数
Curriculum Learning Reward Functions

每个阶段有不同的奖励设计，逐步提高难度
"""

import math
from typing import Dict, Tuple
from core.world import WorldState, EntityState
from config.agent_config import AgentConfig


class Stage1HunterReward:
    """阶段1: 猎人对静止猎物 - 超高捕食奖励 + 增强密集奖励 + 惩罚静止"""

    def __init__(self):
        # 加载配置
        self.agent_config = AgentConfig()
        # 超高捕食奖励
        self.predation_reward = 200.0

        # 增强接近奖励 (3x stronger - 5.0 → 15.0)
        self.approach_scale = 15.0
        self.max_approach_distance = 200.0

        # 减弱方向奖励 (15.0 → 8.0)
        self.direction_scale = 8.0

        # 移动奖励和静止惩罚 (新增强化)
        self.min_speed_threshold = 10.0  # 提高: 3.0 → 10.0 (鼓励快速移动)
        self.movement_reward_scale = 2.0  # 移动奖励系数
        self.stationary_penalty = -3.0  # 静止惩罚 (新增)
        self.low_speed_penalty_scale = 2.0  # 低速惩罚系数 (新增)

        # 减弱转向奖励 (5.0 → 2.0)
        self.turn_reward_scale = 2.0  # 转向奖励系数

        # 增强追击加成 (3.0 → 10.0)
        self.chase_bonus_scale = 10.0

        # 基础惩罚
        self.step_penalty = -0.05

        # 新增: 距离进度追踪
        self.prev_distances = {}  # hunter_id -> min_distance
        self.progress_reward_scale = 10.0  # 接近进度奖励

        # 记录上一帧位置
        self.prev_positions = {}
        self.prev_speeds = {}  # 记录上一帧速度
        self.prev_angles = {}  # 新增: 记录上一帧角度

        # 新增: 持续追击时间追踪
        self.chase_streak = {}  # hunter_id -> consecutive steps of closing distance
        self.max_chase_multiplier = 3.0  # 最大追击倍数
        self.chase_buildup_steps = 10  # 10步达到最大倍数

        # 新增: 非意图性行为惩罚
        self.wrong_direction_penalty = -5.0  # 远离目标的惩罚
        self.backward_movement_penalty = -3.0  # 背向移动惩罚
        self.ineffective_chase_penalty = -2.0  # 无效追击惩罚 (有速度但不接近)

    def compute_reward(self, hunter: EntityState, prev_hunter: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算阶段1猎人奖励"""
        reward = 0.0

        # 1. 基础存活惩罚
        reward += self.step_penalty

        # 2. 捕食奖励 (检查是否新捕食)
        prev_prey_count = sum(1 for e in prev_world.entities if e.type == 'prey')
        curr_prey_count = sum(1 for e in curr_world.entities if e.type == 'prey')
        if curr_prey_count < prev_prey_count:
            # 有猎物死亡，假设是当前猎人捕食的
            reward += self.predation_reward
            print(f"[阶段1] 猎人 {hunter.id} 成功捕食! 奖励: {self.predation_reward:.1f}")

        # 3. 找到最近的猎物
        closest_prey = None
        min_distance = float('inf')
        for entity in curr_world.entities:
            if entity.type == 'prey':
                dx = entity.x - hunter.x
                dy = entity.y - hunter.y
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_prey = entity

        if closest_prey is None:
            return reward  # 没有猎物了

        # 4. 接近奖励 (距离越近奖励越高)
        if min_distance < self.max_approach_distance:
            approach_reward = self.approach_scale * (1.0 - min_distance / self.max_approach_distance)
            reward += approach_reward

        # 4.5 新增: 距离进度奖励 (奖励接近猎物的进度)
        prev_distance = self.prev_distances.get(hunter.id)
        if prev_distance is not None:
            distance_progress = prev_distance - min_distance  # 正值表示接近
            if distance_progress > 0:  # 成功接近
                progress_reward = self.progress_reward_scale * min(distance_progress / 10.0, 1.0)
                reward += progress_reward
            elif distance_progress < -5.0:  # 严重远离，轻微惩罚
                reward -= 0.5

        # 更新距离记录
        self.prev_distances[hunter.id] = min_distance

        # 5. 方向对齐奖励
        dx = closest_prey.x - hunter.x
        dy = closest_prey.y - hunter.y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        alignment = math.cos(angle_diff)

        if alignment > 0.5:  # 大致朝向目标
            direction_reward = self.direction_scale * alignment
            reward += direction_reward

        # 6. 移动奖励与静止惩罚 (强化版)
        speed_ratio = hunter.speed / self.agent_config.HUNTER_SPEED_MAX  # 归一化速度

        if hunter.speed < 2.0:  # 几乎静止
            # 静止惩罚
            reward += self.stationary_penalty
        elif hunter.speed < self.min_speed_threshold:  # 低速
            # 低速惩罚 (线性惩罚)
            low_speed_penalty = -self.low_speed_penalty_scale * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:  # 高速移动
            # 移动奖励 (速度越快奖励越高)
            movement_reward = self.movement_reward_scale * speed_ratio
            reward += movement_reward

        # 7. 增强追击加成 + 持续追击时间奖励
        prev_distance = self.prev_distances.get(hunter.id)
        current_streak = self.chase_streak.get(hunter.id, 0)

        # 判断是否在持续接近
        if prev_distance is not None:
            distance_change = prev_distance - min_distance
            if distance_change > 0:  # 正在接近
                # 增加连击计数
                current_streak = min(current_streak + 1, self.chase_buildup_steps)
            else:
                # 未接近，重置连击
                current_streak = max(0, current_streak - 1)

        self.chase_streak[hunter.id] = current_streak

        # 计算追击倍数 (随持续时间线性增长)
        chase_multiplier = 1.0 + (self.max_chase_multiplier - 1.0) * (current_streak / self.chase_buildup_steps)

        # 强化追击加成: 高速 + 方向对齐 + 持续时间
        if closest_prey is not None and hunter.speed > self.min_speed_threshold:
            if alignment > 0.6:  # 降低阈值 0.7 → 0.6
                # 基础追击奖励
                base_chase = self.chase_bonus_scale * speed_ratio * alignment
                # 应用时间倍数
                chase_bonus = base_chase * chase_multiplier
                reward += chase_bonus

                # 调试输出 (降低阈值到3步，增加输出频率)
                if current_streak >= 3 and current_streak % 2 == 0:  # 3,4,6,8,10步时打印
                    print(f"[追击连击] {hunter.id}: {current_streak}步 × {chase_multiplier:.2f}倍 = +{chase_bonus:.2f}")
            else:
                # 非意图性追击惩罚: 高速移动但方向不对
                misalignment_penalty = self.backward_movement_penalty * speed_ratio * (1.0 - alignment)
                reward += misalignment_penalty
                if abs(misalignment_penalty) > 1.0 and hunter.speed > 15.0:
                    print(f"[背向惩罚] {hunter.id}: alignment={alignment:.2f}, penalty={misalignment_penalty:.2f}")

        # 7.5 新增: 远离目标惩罚 (高速但距离增加)
        if prev_distance is not None and closest_prey is not None:
            distance_change = prev_distance - min_distance
            if distance_change < -3.0 and hunter.speed > self.min_speed_threshold:
                # 高速远离目标，强烈惩罚
                fleeing_penalty = self.wrong_direction_penalty * min(abs(distance_change) / 10.0, 1.0)
                reward += fleeing_penalty
                if abs(fleeing_penalty) > 2.0:
                    print(f"[远离惩罚] {hunter.id}: 远离{abs(distance_change):.1f}px, penalty={fleeing_penalty:.2f}")

        # 7.6 新增: 无效追击惩罚 (中速但长时间不接近)
        if closest_prey is not None and hunter.speed > 5.0:
            if current_streak == 0 and prev_distance is not None:
                # 有一定速度，但连击为0，说明没有持续接近
                distance_change = prev_distance - min_distance
                if abs(distance_change) < 1.0:  # 距离几乎没变化
                    ineffective_penalty = self.ineffective_chase_penalty * (hunter.speed / 50.0)
                    reward += ineffective_penalty

        # 8. 减弱转向奖励 (奖励朝向目标转向)
        if closest_prey is not None:
            prev_angle = self.prev_angles.get(hunter.id)
            if prev_angle is not None:
                # 计算角度变化
                angle_change = hunter.angle - prev_angle
                # 归一化到 [-π, π]
                while angle_change > math.pi:
                    angle_change -= 2 * math.pi
                while angle_change < -math.pi:
                    angle_change += 2 * math.pi

                # 计算目标方向
                dx = closest_prey.x - hunter.x
                dy = closest_prey.y - hunter.y
                target_angle = math.atan2(dy, dx)

                # 计算前一帧到目标的角度差
                prev_angle_diff = target_angle - prev_angle
                while prev_angle_diff > math.pi:
                    prev_angle_diff -= 2 * math.pi
                while prev_angle_diff < -math.pi:
                    prev_angle_diff += 2 * math.pi

                # 计算当前到目标的角度差
                curr_angle_diff = target_angle - hunter.angle
                while curr_angle_diff > math.pi:
                    curr_angle_diff -= 2 * math.pi
                while curr_angle_diff < -math.pi:
                    curr_angle_diff += 2 * math.pi

                # 判断转向是否朝向目标 (角度差绝对值减小)
                if abs(curr_angle_diff) < abs(prev_angle_diff):
                    # 转向正确，给予奖励
                    turn_progress = abs(prev_angle_diff) - abs(curr_angle_diff)
                    turn_reward = self.turn_reward_scale * min(turn_progress / 0.3, 1.0)  # 归一化到0.3弧度
                    reward += turn_reward

        # 更新位置记录
        self.prev_positions[hunter.id] = (hunter.x, hunter.y)
        self.prev_speeds[hunter.id] = hunter.speed
        self.prev_angles[hunter.id] = hunter.angle

        return reward


class Stage2HunterReward:
    """阶段2: 猎人对脚本猎物 - 增加协作奖励 + 惩罚静止"""

    def __init__(self):
        # 加载配置
        self.agent_config = AgentConfig()

        # 捕食奖励 (降低一些)
        self.predation_reward = 150.0

        # 增强接近奖励 (Stage2也增强)
        self.approach_scale = 12.0  # 提高: 4.0 → 12.0
        self.max_approach_distance = 200.0

        # 减弱方向奖励 (12.0 → 6.0)
        self.direction_scale = 6.0

        # 协作奖励
        self.cooperation_scale = 2.0
        self.cooperation_distance = 150.0

        # 移动奖励和静止惩罚
        self.min_speed_threshold = 10.0  # 提高: 5.0 → 10.0
        self.movement_reward_scale = 2.0
        self.stationary_penalty = -3.0
        self.low_speed_penalty_scale = 2.0

        # 减弱转向奖励 (5.0 → 2.0)
        self.turn_reward_scale = 2.0

        # 增强追击加成 (3.0 → 10.0)
        self.chase_bonus_scale = 10.0

        # 基础惩罚
        self.step_penalty = -0.1

        self.prev_positions = {}
        self.prev_distances = {}  # 新增距离追踪
        self.prev_angles = {}  # 新增: 记录上一帧角度

        # 新增: 持续追击时间追踪
        self.chase_streak = {}
        self.max_chase_multiplier = 3.0
        self.chase_buildup_steps = 10

        # 新增: 非意图性行为惩罚 (与Stage1相同)
        self.wrong_direction_penalty = -5.0
        self.backward_movement_penalty = -3.0
        self.ineffective_chase_penalty = -2.0

    def compute_reward(self, hunter: EntityState, prev_hunter: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算阶段2猎人奖励"""
        reward = 0.0

        # 1. 基础存活惩罚
        reward += self.step_penalty

        # 2. 捕食奖励
        prev_prey_count = sum(1 for e in prev_world.entities if e.type == 'prey')
        curr_prey_count = sum(1 for e in curr_world.entities if e.type == 'prey')
        if curr_prey_count < prev_prey_count:
            reward += self.predation_reward
            print(f"[阶段2] 猎人 {hunter.id} 成功捕食! 奖励: {self.predation_reward:.1f}")

        # 3. 找到最近的猎物
        closest_prey = None
        min_distance = float('inf')
        for entity in curr_world.entities:
            if entity.type == 'prey':
                dx = entity.x - hunter.x
                dy = entity.y - hunter.y
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_prey = entity

        if closest_prey is None:
            return reward

        # 4. 接近奖励
        if min_distance < self.max_approach_distance:
            approach_reward = self.approach_scale * (1.0 - min_distance / self.max_approach_distance)
            reward += approach_reward

        # 5. 方向对齐奖励
        dx = closest_prey.x - hunter.x
        dy = closest_prey.y - hunter.y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        alignment = math.cos(angle_diff)

        if alignment > 0.7:
            direction_reward = self.direction_scale * alignment
            reward += direction_reward

        # 6. 协作奖励 (新增!) - 奖励多个猎人围攻同一猎物
        other_hunters_nearby = 0
        for entity in curr_world.entities:
            if entity.type == 'hunter' and entity.id != hunter.id:
                # 检查这个猎人是否也在追同一个猎物
                dx_other = closest_prey.x - entity.x
                dy_other = closest_prey.y - entity.y
                distance_other = math.sqrt(dx_other**2 + dy_other**2)

                if distance_other < self.cooperation_distance:
                    other_hunters_nearby += 1

        if other_hunters_nearby > 0:
            cooperation_reward = self.cooperation_scale * other_hunters_nearby
            reward += cooperation_reward

        # 7. 移动奖励与静止惩罚 (与Stage1一致)
        speed_ratio = hunter.speed / self.agent_config.HUNTER_SPEED_MAX

        if hunter.speed < 2.0:  # 静止
            reward += self.stationary_penalty
        elif hunter.speed < self.min_speed_threshold:  # 低速
            low_speed_penalty = -self.low_speed_penalty_scale * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:  # 高速
            movement_reward = self.movement_reward_scale * speed_ratio
            reward += movement_reward

        # 8. 增强追击加成 + 持续追击时间奖励
        prev_distance = self.prev_distances.get(hunter.id)
        current_streak = self.chase_streak.get(hunter.id, 0)

        # 判断是否在持续接近
        if prev_distance is not None and closest_prey is not None:
            distance_change = prev_distance - min_distance
            if distance_change > 0:  # 正在接近
                current_streak = min(current_streak + 1, self.chase_buildup_steps)
                # 距离进度奖励
                progress_reward = 10.0 * min(distance_change / 10.0, 1.0)
                reward += progress_reward
            else:
                current_streak = max(0, current_streak - 1)

        self.chase_streak[hunter.id] = current_streak

        if closest_prey is not None:
            self.prev_distances[hunter.id] = min_distance

        # 计算追击倍数
        chase_multiplier = 1.0 + (self.max_chase_multiplier - 1.0) * (current_streak / self.chase_buildup_steps)

        # 强化追击加成
        if closest_prey is not None and hunter.speed > self.min_speed_threshold:
            if alignment > 0.6:
                base_chase = self.chase_bonus_scale * speed_ratio * alignment
                chase_bonus = base_chase * chase_multiplier
                reward += chase_bonus

                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[追击连击] {hunter.id}: {current_streak}步 × {chase_multiplier:.2f}倍 = +{chase_bonus:.2f}")
            else:
                # 非意图性追击惩罚
                misalignment_penalty = self.backward_movement_penalty * speed_ratio * (1.0 - alignment)
                reward += misalignment_penalty

        # 8.5 新增: 远离目标惩罚
        if prev_distance is not None and closest_prey is not None:
            distance_change = prev_distance - min_distance
            if distance_change < -3.0 and hunter.speed > self.min_speed_threshold:
                fleeing_penalty = self.wrong_direction_penalty * min(abs(distance_change) / 10.0, 1.0)
                reward += fleeing_penalty

        # 8.6 新增: 无效追击惩罚
        if closest_prey is not None and hunter.speed > 5.0:
            if current_streak == 0 and prev_distance is not None:
                distance_change = prev_distance - min_distance
                if abs(distance_change) < 1.0:
                    ineffective_penalty = self.ineffective_chase_penalty * (hunter.speed / 50.0)
                    reward += ineffective_penalty

        # 9. 减弱转向奖励 (奖励朝向目标转向)
        if closest_prey is not None:
            prev_angle = self.prev_angles.get(hunter.id)
            if prev_angle is not None:
                # 计算目标方向
                dx = closest_prey.x - hunter.x
                dy = closest_prey.y - hunter.y
                target_angle = math.atan2(dy, dx)

                # 计算前一帧到目标的角度差
                prev_angle_diff = target_angle - prev_angle
                while prev_angle_diff > math.pi:
                    prev_angle_diff -= 2 * math.pi
                while prev_angle_diff < -math.pi:
                    prev_angle_diff += 2 * math.pi

                # 计算当前到目标的角度差
                curr_angle_diff = target_angle - hunter.angle
                while curr_angle_diff > math.pi:
                    curr_angle_diff -= 2 * math.pi
                while curr_angle_diff < -math.pi:
                    curr_angle_diff += 2 * math.pi

                # 判断转向是否朝向目标 (角度差绝对值减小)
                if abs(curr_angle_diff) < abs(prev_angle_diff):
                    # 转向正确，给予奖励
                    turn_progress = abs(prev_angle_diff) - abs(curr_angle_diff)
                    turn_reward = self.turn_reward_scale * min(turn_progress / 0.3, 1.0)
                    reward += turn_reward

        self.prev_positions[hunter.id] = (hunter.x, hunter.y)
        self.prev_angles[hunter.id] = hunter.angle

        return reward


class Stage3PreyReward:
    """阶段3: 猎物训练 - 强逃跑与集群奖励 + 惩罚静止"""

    def __init__(self):
        # 加载配置
        self.agent_config = AgentConfig()

        # 存活奖励
        self.survival_bonus = 0.5

        # 增强逃跑奖励
        self.escape_scale = 15.0  # 提高: 5.0 → 15.0 (大幅增强)
        self.danger_distance = 200.0  # 扩大: 150.0 → 200.0 (更早感知危险)

        # 集群奖励
        self.herd_scale = 3.0
        self.herd_distance = 100.0

        # 增强躲避奖励
        self.evasion_bonus = 10.0  # 提高: 5.0 → 10.0

        # 移动奖励和静止惩罚
        self.min_speed_threshold = 10.0  # 提高: 5.0 → 10.0
        self.movement_reward_scale = 2.0  # 新增
        self.stationary_penalty = -3.0  # 新增
        self.low_speed_penalty_scale = 2.0  # 新增

        # 被捕食惩罚
        self.death_penalty = -50.0

        # 减弱逃跑方向奖励 (10.0 → 5.0)
        self.flee_direction_scale = 5.0

        # 减弱转向奖励 (5.0 → 2.0)
        self.turn_reward_scale = 2.0

        # 增强逃跑加成 (5.0 → 12.0)
        self.escape_bonus_scale = 12.0

        self.prev_positions = {}
        self.prev_min_distances = {}
        self.prev_angles = {}  # 新增: 记录上一帧角度

        # 新增: 持续逃跑时间追踪
        self.escape_streak = {}
        self.max_escape_multiplier = 3.0
        self.escape_buildup_steps = 10

        # 新增: 非意图性行为惩罚 (猎物版本)
        self.approach_hunter_penalty = -8.0  # 接近猎人惩罚 (比猎人的-5.0更强)
        self.facing_hunter_penalty = -4.0  # 面向猎人惩罚
        self.ineffective_escape_penalty = -3.0  # 无效逃跑惩罚

    def compute_reward(self, prey: EntityState, prev_prey: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算阶段3猎物奖励"""
        reward = 0.0

        # 1. 基础存活奖励
        reward += self.survival_bonus

        # 2. 找到最近的猎人
        closest_hunter = None
        min_distance = float('inf')
        for entity in curr_world.entities:
            if entity.type == 'hunter':
                dx = entity.x - prey.x
                dy = entity.y - prey.y
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_hunter = entity

        if closest_hunter is None:
            return reward  # 没有猎人了(不太可能)

        # 3. 逃跑奖励 (距离越远奖励越高)
        if min_distance < self.danger_distance:
            danger_level = 1.0 - (min_distance / self.danger_distance)
            escape_reward = self.escape_scale * danger_level
            reward += escape_reward

        # 4. 躲避奖励 (距离增加)
        prev_min_distance = self.prev_min_distances.get(prey.id)
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance
            if distance_change > 0:  # 拉开距离
                evasion_reward = self.evasion_bonus * (distance_change / 10.0)
                reward += min(evasion_reward, 2.0)  # 限制上限

        # 5. 集群奖励 (新增!) - 奖励与其他猎物聚集
        nearby_prey = 0
        for entity in curr_world.entities:
            if entity.type == 'prey' and entity.id != prey.id:
                dx = entity.x - prey.x
                dy = entity.y - prey.y
                distance = math.sqrt(dx**2 + dy**2)
                if distance < self.herd_distance:
                    nearby_prey += 1

        if nearby_prey > 0:
            herd_reward = self.herd_scale * math.sqrt(nearby_prey)
            reward += herd_reward

        # 6. 增强逃跑方向奖励 (背离猎人)
        dx = closest_hunter.x - prey.x
        dy = closest_hunter.y - prey.y
        threat_angle = math.atan2(dy, dx)

        # 计算逃跑方向对齐度
        angle_diff = abs(prey.angle - threat_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # 计算逃跑对齐度 (背离猎人的程度)
        flee_alignment = math.cos(angle_diff + math.pi)  # 翻转cos

        # 奖励背离猎人的方向 (角度差接近180度)
        if angle_diff > math.pi / 2:
            if flee_alignment > 0:
                flee_reward = self.flee_direction_scale * flee_alignment
                reward += flee_reward

        # 7. 移动奖励与静止惩罚 (强制猎物移动)
        speed_ratio = prey.speed / self.agent_config.PREY_SPEED_MAX

        if prey.speed < 2.0:  # 静止
            reward += self.stationary_penalty
        elif prey.speed < self.min_speed_threshold:  # 低速
            low_speed_penalty = -self.low_speed_penalty_scale * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:  # 高速
            movement_reward = self.movement_reward_scale * speed_ratio
            reward += movement_reward

        # 8. 增强逃跑加成 + 持续逃跑时间奖励
        prev_min_distance = self.prev_min_distances.get(prey.id)
        current_streak = self.escape_streak.get(prey.id, 0)

        # 判断是否在持续远离
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance
            if distance_change > 0:  # 正在远离
                current_streak = min(current_streak + 1, self.escape_buildup_steps)
            else:
                current_streak = max(0, current_streak - 1)

        self.escape_streak[prey.id] = current_streak

        # 计算逃跑倍数
        escape_multiplier = 1.0 + (self.max_escape_multiplier - 1.0) * (current_streak / self.escape_buildup_steps)

        # 强化逃跑加成: 在危险时高速逃跑
        if min_distance < self.danger_distance and prey.speed > self.min_speed_threshold:
            if angle_diff > math.pi / 2:  # 正在逃跑
                danger_level = 1.0 - (min_distance / self.danger_distance)
                base_escape = self.escape_bonus_scale * speed_ratio * flee_alignment * danger_level
                escape_bonus = base_escape * escape_multiplier
                reward += escape_bonus

                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[逃跑连击] {prey.id}: {current_streak}步 × {escape_multiplier:.2f}倍 = +{escape_bonus:.2f}")
            else:
                # 非意图性逃跑惩罚: 在危险区域内但朝向猎人
                if min_distance < self.danger_distance:
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    facing_penalty = self.facing_hunter_penalty * speed_ratio * danger_level * flee_alignment
                    reward += facing_penalty
                    if abs(facing_penalty) > 1.5:
                        print(f"[面向惩罚] {prey.id}: 距离={min_distance:.1f}, penalty={facing_penalty:.2f}")

        # 8.5 新增: 接近猎人惩罚
        if prev_min_distance is not None and closest_hunter is not None:
            distance_change = min_distance - prev_min_distance
            if distance_change < -2.0 and prey.speed > self.min_speed_threshold:
                # 高速接近猎人，重度惩罚
                if min_distance < self.danger_distance:
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    approach_penalty = self.approach_hunter_penalty * min(abs(distance_change) / 10.0, 1.0) * danger_level
                    reward += approach_penalty
                    if abs(approach_penalty) > 2.0:
                        print(f"[接近惩罚] {prey.id}: 接近{abs(distance_change):.1f}px, penalty={approach_penalty:.2f}")

        # 8.6 新增: 无效逃跑惩罚 (危险区域内但不远离)
        if min_distance < self.danger_distance and prey.speed > 5.0:
            if current_streak == 0 and prev_min_distance is not None:
                distance_change = min_distance - prev_min_distance
                if abs(distance_change) < 1.0:  # 距离几乎没变化
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    ineffective_penalty = self.ineffective_escape_penalty * (prey.speed / 45.0) * danger_level
                    reward += ineffective_penalty

        # 9. 减弱转向奖励 (奖励背离猎人转向)
        if closest_hunter is not None:
            prev_angle = self.prev_angles.get(prey.id)
            if prev_angle is not None:
                # 计算威胁方向 (猎人位置)
                dx = closest_hunter.x - prey.x
                dy = closest_hunter.y - prey.y
                threat_angle = math.atan2(dy, dx)

                # 计算前一帧与威胁方向的角度差
                prev_angle_diff = prev_angle - threat_angle
                while prev_angle_diff > math.pi:
                    prev_angle_diff -= 2 * math.pi
                while prev_angle_diff < -math.pi:
                    prev_angle_diff += 2 * math.pi

                # 计算当前与威胁方向的角度差
                curr_angle_diff = prey.angle - threat_angle
                while curr_angle_diff > math.pi:
                    curr_angle_diff -= 2 * math.pi
                while curr_angle_diff < -math.pi:
                    curr_angle_diff += 2 * math.pi

                # 计算前一帧和当前帧到"背离方向"(180度)的接近度
                # 背离方向是threat_angle + π
                ideal_flee_diff_prev = abs(abs(prev_angle_diff) - math.pi)
                ideal_flee_diff_curr = abs(abs(curr_angle_diff) - math.pi)

                # 如果当前更接近背离方向，给予奖励
                if ideal_flee_diff_curr < ideal_flee_diff_prev:
                    turn_progress = ideal_flee_diff_prev - ideal_flee_diff_curr
                    turn_reward = self.turn_reward_scale * min(turn_progress / 0.3, 1.0)
                    reward += turn_reward

        # 更新记录
        self.prev_positions[prey.id] = (prey.x, prey.y)
        self.prev_min_distances[prey.id] = min_distance
        self.prev_angles[prey.id] = prey.angle

        return reward


class Stage4JointReward:
    """阶段4: 联合微调 - 使用V2奖励但降低学习率"""

    def __init__(self):
        # 导入V2奖励函数
        from rl_env import EnhancedRewardFunctionV2
        self.v2_reward = EnhancedRewardFunctionV2()

    def compute_reward(self, entity: EntityState, prev_entity: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算阶段4奖励 - 直接使用V2"""
        return self.v2_reward.compute_reward(entity, prev_entity, prev_world, curr_world)


class CurriculumRewardFunction:
    """统一的课程学习奖励函数接口"""

    def __init__(self, stage: str):
        """
        Args:
            stage: "stage1" | "stage2" | "stage3" | "stage4"
        """
        self.stage = stage

        if stage == "stage1":
            self.hunter_reward = Stage1HunterReward()
            self.prey_reward = None  # 阶段1猎物不学习
        elif stage == "stage2":
            self.hunter_reward = Stage2HunterReward()
            self.prey_reward = None  # 阶段2猎物脚本控制
        elif stage == "stage3":
            self.hunter_reward = None  # 阶段3猎人冻结
            self.prey_reward = Stage3PreyReward()
        elif stage == "stage4":
            self.joint_reward = Stage4JointReward()
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def reset(self):
        """重置奖励函数内部状态"""
        if self.stage == "stage4":
            if hasattr(self.joint_reward.v2_reward, 'reset'):
                self.joint_reward.v2_reward.reset()
        else:
            # 重置各阶段奖励函数的内部状态
            if hasattr(self, 'hunter_reward') and self.hunter_reward is not None:
                self.hunter_reward.prev_positions = {}
                if hasattr(self.hunter_reward, 'prev_distances'):
                    self.hunter_reward.prev_distances = {}
                if hasattr(self.hunter_reward, 'prev_angles'):
                    self.hunter_reward.prev_angles = {}
            if hasattr(self, 'prey_reward') and self.prey_reward is not None:
                self.prey_reward.prev_positions = {}
                self.prey_reward.prev_min_distances = {}
                if hasattr(self.prey_reward, 'prev_angles'):
                    self.prey_reward.prev_angles = {}

    def compute_reward(self, entity: EntityState, prev_entity: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算奖励"""
        if self.stage == "stage4":
            return self.joint_reward.compute_reward(entity, prev_entity, prev_world, curr_world)

        if entity.type == 'hunter':
            if self.hunter_reward is None:
                return 0.0  # 阶段3猎人冻结，不给奖励
            return self.hunter_reward.compute_reward(entity, prev_entity, prev_world, curr_world)
        else:  # prey
            if self.prey_reward is None:
                return 0.0  # 阶段1/2猎物不学习，不给奖励
            return self.prey_reward.compute_reward(entity, prev_entity, prev_world, curr_world)

    def compute_population_rewards(self, world: WorldState) -> Tuple[float, float, str]:
        """计算种群平衡奖励 (课程学习中不使用)"""
        return 0.0, 0.0, ""
