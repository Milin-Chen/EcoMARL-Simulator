"""
HPO增强版课程学习奖励函数
Curriculum Reward Functions with HPO Enhancements
"""

import math
from typing import Dict, Tuple
from core.world import WorldState, EntityState
from .hpo_enhancements import HPORewardEnhancer


class Stage1HunterRewardHPO:
    """阶段1: 猎人对静止猎物 - HPO增强版"""

    def __init__(self, total_steps: int = 45000, enable_hpo: bool = True):
        """
        Args:
            total_steps: 该阶段总训练步数
            enable_hpo: 是否启用HPO增强
        """
        # 基础奖励参数
        self.predation_reward = 200.0
        self.approach_scale = 15.0
        self.max_approach_distance = 200.0
        self.direction_scale = 15.0
        self.min_speed_threshold = 10.0
        self.movement_reward_scale = 2.0
        self.stationary_penalty = -3.0
        self.low_speed_penalty_scale = 2.0
        self.turn_reward_scale = 5.0
        self.step_penalty = -0.05
        self.progress_reward_scale = 10.0

        # HPO增强器
        self.enable_hpo = enable_hpo
        if enable_hpo:
            self.hpo_enhancer = HPORewardEnhancer(
                total_steps=total_steps,
                enable_adaptive=True,
                enable_balancing=True,
                enable_distance=True,
            )
        else:
            self.hpo_enhancer = None

        # 状态追踪
        self.prev_distances = {}
        self.prev_positions = {}
        self.prev_speeds = {}
        self.prev_angles = {}

        # 新增: 持续追击时间追踪 (与普通版本一致)
        self.chase_streak = {}  # hunter_id -> consecutive steps of closing distance
        self.max_chase_multiplier = 3.0  # 最大追击倍数
        self.chase_buildup_steps = 10  # 10步达到最大倍数

    def compute_reward(
        self,
        hunter: EntityState,
        prev_hunter: EntityState,
        prev_world: WorldState,
        curr_world: WorldState
    ) -> float:
        """计算阶段1猎人奖励 (HPO增强)"""
        reward = 0.0

        # 获取HPO权重 (如果启用)
        if self.hpo_enhancer:
            weights = self.hpo_enhancer.get_reward_weights()
            hunter_mult, _ = self.hpo_enhancer.get_balance_multipliers()
        else:
            # 默认权重
            weights = {
                'movement': self.movement_reward_scale,
                'turn': self.turn_reward_scale,
                'direction': self.direction_scale,
                'approach': self.approach_scale,
                'capture': self.predation_reward,
                'stationary': self.stationary_penalty,
                'low_speed': self.low_speed_penalty_scale,
            }
            hunter_mult = 1.0

        # 1. 基础存活惩罚
        reward += self.step_penalty

        # 2. 捕食奖励
        prev_prey_count = sum(1 for e in prev_world.entities if e.type == 'prey')
        curr_prey_count = sum(1 for e in curr_world.entities if e.type == 'prey')
        if curr_prey_count < prev_prey_count:
            reward += weights['capture']
            print(f"[Stage1-HPO] 猎人 {hunter.id} 捕食成功! 奖励: {weights['capture']:.1f}")

            # 更新对抗统计
            if self.hpo_enhancer:
                self.hpo_enhancer.update_outcome('capture')

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

        # 4. 接近奖励 (使用动态权重)
        if min_distance < self.max_approach_distance:
            approach_reward = weights['approach'] * \
                            (1.0 - min_distance / self.max_approach_distance)
            reward += approach_reward

        # 4.5 HPO距离进度奖励
        if self.hpo_enhancer:
            distance_progress_reward = self.hpo_enhancer.compute_distance_progress_reward(
                hunter.id, 'hunter', min_distance, scale=self.progress_reward_scale
            )
            reward += distance_progress_reward

        # 5. 方向对齐奖励 (使用动态权重)
        dx = closest_prey.x - hunter.x
        dy = closest_prey.y - hunter.y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        alignment = math.cos(angle_diff)

        if alignment > 0.5:
            direction_reward = weights['direction'] * alignment
            reward += direction_reward

        # 6. 移动奖励与静止惩罚 (使用动态权重)
        speed_ratio = hunter.speed / 50.0

        if hunter.speed < 2.0:
            reward += weights['stationary']
        elif hunter.speed < self.min_speed_threshold:
            low_speed_penalty = -weights['low_speed'] * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:
            movement_reward = weights['movement'] * speed_ratio
            reward += movement_reward

        # 7. 增强追击加成 + 持续追击时间奖励 (与普通版本一致)
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
                base_chase = 10.0 * speed_ratio * alignment  # 使用固定scale，HPO会通过weights调整
                # 应用时间倍数
                chase_bonus = base_chase * chase_multiplier
                reward += chase_bonus

                # 调试输出
                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[HPO追击连击] {hunter.id}: {current_streak}步 × {chase_multiplier:.2f}倍 = +{chase_bonus:.2f}")

        # 8. 转向奖励 (使用动态权重)
        if closest_prey is not None:
            prev_angle = self.prev_angles.get(hunter.id)
            if prev_angle is not None:
                dx = closest_prey.x - hunter.x
                dy = closest_prey.y - hunter.y
                target_angle = math.atan2(dy, dx)

                prev_angle_diff = target_angle - prev_angle
                while prev_angle_diff > math.pi:
                    prev_angle_diff -= 2 * math.pi
                while prev_angle_diff < -math.pi:
                    prev_angle_diff += 2 * math.pi

                curr_angle_diff = target_angle - hunter.angle
                while curr_angle_diff > math.pi:
                    curr_angle_diff -= 2 * math.pi
                while curr_angle_diff < -math.pi:
                    curr_angle_diff += 2 * math.pi

                if abs(curr_angle_diff) < abs(prev_angle_diff):
                    turn_progress = abs(prev_angle_diff) - abs(curr_angle_diff)
                    turn_reward = weights['turn'] * min(turn_progress / 0.3, 1.0)
                    reward += turn_reward

        # 更新记录
        self.prev_positions[hunter.id] = (hunter.x, hunter.y)
        self.prev_speeds[hunter.id] = hunter.speed
        self.prev_angles[hunter.id] = hunter.angle
        self.prev_distances[hunter.id] = min_distance

        # 应用对抗平衡系数
        reward *= hunter_mult

        return reward


class Stage3PreyRewardHPO:
    """阶段3: 猎物训练 - HPO增强版"""

    def __init__(self, total_steps: int = 45000, enable_hpo: bool = True):
        """
        Args:
            total_steps: 该阶段总训练步数
            enable_hpo: 是否启用HPO增强
        """
        # 基础奖励参数
        self.survival_bonus = 0.5
        self.escape_scale = 15.0
        self.danger_distance = 200.0
        self.herd_scale = 3.0
        self.herd_distance = 100.0
        self.evasion_bonus = 10.0
        self.min_speed_threshold = 10.0
        self.movement_reward_scale = 2.0
        self.stationary_penalty = -3.0
        self.low_speed_penalty_scale = 2.0
        self.death_penalty = -50.0
        self.flee_direction_scale = 10.0
        self.turn_reward_scale = 5.0

        # HPO增强器
        self.enable_hpo = enable_hpo
        if enable_hpo:
            self.hpo_enhancer = HPORewardEnhancer(
                total_steps=total_steps,
                enable_adaptive=True,
                enable_balancing=True,
                enable_distance=True,
            )
        else:
            self.hpo_enhancer = None

        # 状态追踪
        self.prev_positions = {}
        self.prev_min_distances = {}
        self.prev_angles = {}

        # 新增: 持续逃跑时间追踪 (与普通版本一致)
        self.escape_streak = {}  # prey_id -> consecutive steps of increasing distance
        self.max_escape_multiplier = 3.0  # 最大逃跑倍数
        self.escape_buildup_steps = 10  # 10步达到最大倍数

    def compute_reward(
        self,
        prey: EntityState,
        prev_prey: EntityState,
        prev_world: WorldState,
        curr_world: WorldState
    ) -> float:
        """计算阶段3猎物奖励 (HPO增强)"""
        reward = 0.0

        # 获取HPO权重
        if self.hpo_enhancer:
            weights = self.hpo_enhancer.get_reward_weights()
            _, prey_mult = self.hpo_enhancer.get_balance_multipliers()
        else:
            weights = {
                'movement': self.movement_reward_scale,
                'turn': self.turn_reward_scale,
                'escape': self.escape_scale,
                'stationary': self.stationary_penalty,
                'low_speed': self.low_speed_penalty_scale,
            }
            prey_mult = 1.0

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
            # 更新逃脱统计
            if self.hpo_enhancer:
                self.hpo_enhancer.update_outcome('escape')
            return reward

        # 3. 逃跑奖励 (使用动态权重)
        if min_distance < self.danger_distance:
            danger_level = 1.0 - (min_distance / self.danger_distance)
            escape_reward = weights['escape'] * danger_level
            reward += escape_reward

        # 4. 躲避奖励 + HPO距离进度
        prev_min_distance = self.prev_min_distances.get(prey.id)
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance
            if distance_change > 0:
                evasion_reward = self.evasion_bonus * (distance_change / 10.0)
                reward += min(evasion_reward, 2.0)

        if self.hpo_enhancer:
            distance_progress_reward = self.hpo_enhancer.compute_distance_progress_reward(
                prey.id, 'prey', min_distance, scale=self.evasion_bonus
            )
            reward += distance_progress_reward

        # 5. 集群奖励
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

        # 6. 逃跑方向奖励
        dx = closest_hunter.x - prey.x
        dy = closest_hunter.y - prey.y
        threat_angle = math.atan2(dy, dx)

        angle_diff = abs(prey.angle - threat_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        if angle_diff > math.pi / 2:
            flee_alignment = math.cos(angle_diff + math.pi)
            if flee_alignment > 0:
                flee_reward = self.flee_direction_scale * flee_alignment
                reward += flee_reward

        # 7. 移动奖励与静止惩罚 (使用动态权重)
        speed_ratio = prey.speed / 45.0

        if prey.speed < 2.0:
            reward += weights['stationary']
        elif prey.speed < self.min_speed_threshold:
            low_speed_penalty = -weights['low_speed'] * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:
            movement_reward = weights['movement'] * speed_ratio
            reward += movement_reward

        # 8. 增强逃跑加成 + 持续逃跑时间奖励 (与普通版本一致)
        prev_min_distance = self.prev_min_distances.get(prey.id)
        current_streak = self.escape_streak.get(prey.id, 0)

        # 判断是否在持续远离
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance  # 正值表示远离
            if distance_change > 0:  # 正在远离
                # 增加连击计数
                current_streak = min(current_streak + 1, self.escape_buildup_steps)
            else:
                # 未远离，重置连击
                current_streak = max(0, current_streak - 1)

        self.escape_streak[prey.id] = current_streak

        # 计算逃跑倍数 (随持续时间线性增长)
        escape_multiplier = 1.0 + (self.max_escape_multiplier - 1.0) * (current_streak / self.escape_buildup_steps)

        # 强化逃跑加成: 高速 + 方向对齐 + 危险程度 + 持续时间
        if min_distance < self.danger_distance and prey.speed > self.min_speed_threshold:
            if angle_diff > math.pi / 2:
                danger_level = 1.0 - (min_distance / self.danger_distance)
                # 基础逃跑奖励
                base_escape = 5.0 * speed_ratio * flee_alignment * danger_level
                # 应用时间倍数
                escape_bonus = base_escape * escape_multiplier
                reward += escape_bonus

                # 调试输出
                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[HPO逃跑连击] {prey.id}: {current_streak}步 × {escape_multiplier:.2f}倍 = +{escape_bonus:.2f}")

        # 9. 转向奖励 (使用动态权重)
        if closest_hunter is not None:
            prev_angle = self.prev_angles.get(prey.id)
            if prev_angle is not None:
                dx = closest_hunter.x - prey.x
                dy = closest_hunter.y - prey.y
                threat_angle = math.atan2(dy, dx)

                prev_angle_diff = prev_angle - threat_angle
                while prev_angle_diff > math.pi:
                    prev_angle_diff -= 2 * math.pi
                while prev_angle_diff < -math.pi:
                    prev_angle_diff += 2 * math.pi

                curr_angle_diff = prey.angle - threat_angle
                while curr_angle_diff > math.pi:
                    curr_angle_diff -= 2 * math.pi
                while curr_angle_diff < -math.pi:
                    curr_angle_diff += 2 * math.pi

                ideal_flee_diff_prev = abs(abs(prev_angle_diff) - math.pi)
                ideal_flee_diff_curr = abs(abs(curr_angle_diff) - math.pi)

                if ideal_flee_diff_curr < ideal_flee_diff_prev:
                    turn_progress = ideal_flee_diff_prev - ideal_flee_diff_curr
                    turn_reward = weights['turn'] * min(turn_progress / 0.3, 1.0)
                    reward += turn_reward

        # 更新记录
        self.prev_positions[prey.id] = (prey.x, prey.y)
        self.prev_min_distances[prey.id] = min_distance
        self.prev_angles[prey.id] = prey.angle

        # 应用对抗平衡系数
        reward *= prey_mult

        return reward


# 便捷函数：创建HPO增强或普通版本
def create_stage1_hunter_reward(total_steps: int = 45000, enable_hpo: bool = True):
    """创建Stage1猎手奖励函数"""
    if enable_hpo:
        return Stage1HunterRewardHPO(total_steps, enable_hpo=True)
    else:
        from rl_env import Stage1HunterReward
        return Stage1HunterReward()


def create_stage3_prey_reward(total_steps: int = 45000, enable_hpo: bool = True):
    """创建Stage3猎物奖励函数"""
    if enable_hpo:
        return Stage3PreyRewardHPO(total_steps, enable_hpo=True)
    else:
        from rl_env import Stage3PreyReward
        return Stage3PreyReward()
