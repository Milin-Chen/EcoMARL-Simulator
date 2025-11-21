"""
HPO增强版课程学习奖励函数
Curriculum Reward Functions with HPO Enhancements
"""

import math
from typing import Dict, Tuple
from core.world import WorldState, EntityState
from .hpo_enhancements import HPORewardEnhancer

# 导入训练日志器
try:
    from ..training_logger import log_penalty, log_event
except ImportError:
    # 如果logger不可用，使用空函数
    def log_penalty(penalty_type: str, value: float = 1.0):
        pass
    def log_event(event_type: str):
        pass


class Stage1HunterRewardHPO:
    """阶段1: 猎人对静止猎物 - HPO增强版"""

    def __init__(self, total_steps: int = 45000, enable_hpo: bool = True):
        """
        Args:
            total_steps: 该阶段总训练步数
            enable_hpo: 是否启用HPO增强
        """
        # 基础奖励参数 (同步标准版: 加强静止惩罚 + 降低距离奖励)
        self.predation_reward = 200.0
        self.approach_scale = 8.0  # 降低: 15.0 → 8.0
        self.max_approach_distance = 200.0
        self.direction_scale = 15.0
        self.min_speed_threshold = 15.0  # 提高: 10.0 → 15.0
        self.movement_reward_scale = 2.0
        self.stationary_penalty = -15.0  # 再次加强: -8.0 → -15.0
        self.low_speed_penalty_scale = 5.0  # 再次加强: 3.0 → 5.0
        self.turn_reward_scale = 0.5  # 修改: 5.0 → 0.5 (防止原地转圈)
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

        # 新增: 目标锁定机制 (防止多目标摇摆)
        self.locked_target = {}  # hunter_id -> prey_id
        self.lock_duration = {}  # hunter_id -> remaining steps
        self.min_lock_steps = 8  # 最少锁定8步
        self.target_switch_penalty = -15.0  # 频繁切换目标的惩罚

        # 新增: 位置抖动检测 (防止原地打转)
        self.position_history = {}  # hunter_id -> [(x, y), ...]
        self.position_history_length = 5  # 追踪5步历史
        self.jitter_penalty = -20.0  # 原地抖动惩罚 (修改: -12.0 → -20.0)
        self.jitter_radius_threshold = 10.0  # 5步内活动半径<10px判定为抖动 (修改: 20.0 → 10.0)

        # 高角速度惩罚 (防止原地快速转向)
        self.high_angular_velocity_penalty = -5.0
        self.angular_velocity_threshold = 0.3

        # 视野丢失惩罚 (防止围绕猎物转圈但不在视野内)
        from config.agent_config import AgentConfig
        self.agent_config = AgentConfig()
        self.vision_loss_penalty = -10.0
        self.vision_check_distance = 200.0
        self.fov_angle = self.agent_config.HUNTER_FOV_DEG * math.pi / 180.0 / 2.0

        # 同类重叠惩罚 (强力防止猎人聚集到同一点)
        self.same_type_overlap_penalty = -30.0  # 基础惩罚 (增强: -15.0 → -30.0, 防止猎人集群)
        self.overlap_penalty_multiplier = 3.0   # 重叠程度倍数 (增强: 2.0 → 3.0)
        self.min_safe_distance_multiplier = 1.5 # 最小安全距离 = (r1+r2) * 1.5 (增强: 1.2 → 1.5)

    def detect_same_type_overlap(self, entity: EntityState, world: WorldState) -> Tuple[int, float]:
        """检测与同类的重叠情况

        Returns:
            overlap_count: 重叠的同类数量
            max_overlap_ratio: 最大重叠程度 [0, 1]
        """
        overlap_count = 0
        max_overlap_ratio = 0.0

        for other in world.entities:
            # 只检测同类，排除自己
            if other.type != entity.type or other.id == entity.id:
                continue

            # 计算距离
            dx = other.x - entity.x
            dy = other.y - entity.y
            distance = math.sqrt(dx**2 + dy**2)

            # 检测重叠 (距离 < 半径和)
            min_distance = entity.radius + other.radius
            if distance < min_distance:
                overlap_count += 1
                # 计算重叠程度 (归一化到0-1)
                overlap_depth = min_distance - distance
                overlap_ratio = overlap_depth / min_distance
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        return overlap_count, max_overlap_ratio

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

            # 捕食成功，清除目标锁定
            if hunter.id in self.locked_target:
                del self.locked_target[hunter.id]
                del self.lock_duration[hunter.id]

        # 2.5 位置抖动检测 (防止原地打转)
        if hunter.id not in self.position_history:
            self.position_history[hunter.id] = []

        self.position_history[hunter.id].append((hunter.x, hunter.y))
        if len(self.position_history[hunter.id]) > self.position_history_length:
            self.position_history[hunter.id].pop(0)

        if len(self.position_history[hunter.id]) >= self.position_history_length:
            positions = self.position_history[hunter.id]
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            max_radius = max(math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions)

            if max_radius < self.jitter_radius_threshold:
                reward += self.jitter_penalty
                log_penalty('jitter', abs(self.jitter_penalty))
                if hunter.speed > 10.0:
                    print(f"[抖动惩罚-HPO] 猎人 {hunter.id} 原地打转! 半径={max_radius:.1f}px")

        # 3. 找到最近的猎物 (优先考虑锁定目标)
        closest_prey = None
        min_distance = float('inf')
        locked_prey = None

        all_prey = [e for e in curr_world.entities if e.type == 'prey']
        if not all_prey:
            return reward

        # 检查锁定目标
        if hunter.id in self.locked_target:
            locked_prey_id = self.locked_target[hunter.id]
            for prey in all_prey:
                if prey.id == locked_prey_id:
                    locked_prey = prey
                    break
            if locked_prey is None:
                del self.locked_target[hunter.id]
                if hunter.id in self.lock_duration:
                    del self.lock_duration[hunter.id]

        # 找到最近的猎物
        for entity in all_prey:
            dx = entity.x - hunter.x
            dy = entity.y - hunter.y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_prey = entity

        # 目标选择逻辑
        if hunter.id in self.lock_duration and self.lock_duration[hunter.id] > 0:
            if locked_prey is not None:
                target_prey = locked_prey
                dx = locked_prey.x - hunter.x
                dy = locked_prey.y - hunter.y
                min_distance = math.sqrt(dx**2 + dy**2)
                self.lock_duration[hunter.id] -= 1
            else:
                target_prey = closest_prey
        else:
            target_prey = closest_prey
            if hunter.id in self.locked_target:
                prev_target_id = self.locked_target[hunter.id]
                if target_prey.id != prev_target_id:
                    reward += self.target_switch_penalty
                    print(f"[目标切换-HPO] 猎人 {hunter.id} 切换目标, 惩罚={self.target_switch_penalty:.1f}")
            self.locked_target[hunter.id] = target_prey.id
            self.lock_duration[hunter.id] = self.min_lock_steps

        closest_prey = target_prey

        # 4. 接近奖励 (必须配合速度! 防止低速刷分)
        if min_distance < self.max_approach_distance:
            approach_factor = 1.0 - min_distance / self.max_approach_distance

            # 速度调制: 低速大幅削减奖励
            if hunter.speed < self.min_speed_threshold:
                speed_mult = 0.2  # 低速只给20%
            else:
                speed_mult = min(hunter.speed / 50.0, 1.0)  # 高速给满额

            approach_reward = weights['approach'] * approach_factor * speed_mult
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

        # 5.5 视野丢失惩罚 (防止围绕猎物转圈但猎物不在视野内)
        if min_distance < self.vision_check_distance:
            if angle_diff > self.fov_angle:
                vision_loss_factor = 1.0 - (min_distance / self.vision_check_distance)
                vision_penalty = self.vision_loss_penalty * vision_loss_factor
                reward += vision_penalty
                if abs(vision_penalty) > 5.0:
                    print(f"[视野丢失HPO] 猎人 {hunter.id} 猎物在{min_distance:.1f}px但不在视野, 惩罚={vision_penalty:.2f}")

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

        # 7. 增强追击加成 + 持续追击时间奖励 (强化版: 要求实质性接近)
        prev_distance = self.prev_distances.get(hunter.id)
        current_streak = self.chase_streak.get(hunter.id, 0)

        # 判断是否在持续接近 (增加最小接近量要求)
        distance_change = 0.0
        if prev_distance is not None:
            distance_change = prev_distance - min_distance
            # 要求每步至少接近0.5px才算有效追击
            if distance_change > 0.5:  # 实质性接近
                # 增加连击计数
                current_streak = min(current_streak + 1, self.chase_buildup_steps)
            else:
                # 未有效接近，快速衰减连击
                current_streak = max(0, current_streak - 2)  # 衰减更快，从-1改为-2

        self.chase_streak[hunter.id] = current_streak

        # 计算追击倍数 (随持续时间线性增长)
        chase_multiplier = 1.0 + (self.max_chase_multiplier - 1.0) * (current_streak / self.chase_buildup_steps)

        # 强化追击加成: 高速 + 方向对齐 + 持续接近 + 接近速度
        if closest_prey is not None and hunter.speed > self.min_speed_threshold:
            if alignment > 0.6 and prev_distance is not None:
                # 计算接近速度因子 (归一化到0-1，接近越快奖励越高)
                approach_factor = 0.0
                if distance_change > 0.5:  # 有效接近
                    # 接近5px以上得到满分，线性插值
                    approach_factor = min(distance_change / 5.0, 1.0)
                else:
                    # 未有效接近，即使方向对齐也只给少量奖励
                    approach_factor = 0.2  # 降低基础奖励

                # 基础追击奖励 (包含接近速度)
                base_chase = 10.0 * speed_ratio * alignment * approach_factor
                # 应用时间倍数
                chase_bonus = base_chase * chase_multiplier
                reward += chase_bonus

                # 调试输出 (显示接近量)
                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[HPO追击连击] {hunter.id}: {current_streak}步 × {chase_multiplier:.2f}倍, 接近{distance_change:.1f}px = +{chase_bonus:.2f}")

        # 8. 转向奖励 (使用动态权重)
        if closest_prey is not None:
            prev_angle = self.prev_angles.get(hunter.id)
            if prev_angle is not None:
                # 高角速度惩罚 (防止原地快速转圈)
                angle_change = hunter.angle - prev_angle
                while angle_change > math.pi:
                    angle_change -= 2 * math.pi
                while angle_change < -math.pi:
                    angle_change += 2 * math.pi

                angular_velocity = abs(angle_change)
                if angular_velocity > self.angular_velocity_threshold:
                    high_angular_penalty = self.high_angular_velocity_penalty * (angular_velocity / math.pi)
                    reward += high_angular_penalty
                    log_penalty('high_angular', abs(high_angular_penalty))
                    if angular_velocity > 0.5:
                        print(f"[高角速度惩罚HPO] 猎人 {hunter.id} 角速度={angular_velocity:.3f}rad, 惩罚={high_angular_penalty:.2f}")

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

        # 9. 同类重叠惩罚 (防止猎人聚集到同一点)
        overlap_count, max_overlap_ratio = self.detect_same_type_overlap(hunter, curr_world)
        if overlap_count > 0:
            # 计算惩罚: 基础惩罚 × 重叠数量 × (1 + 重叠程度加成)
            overlap_penalty = (
                self.same_type_overlap_penalty *
                overlap_count *
                (1.0 + self.overlap_penalty_multiplier * max_overlap_ratio)
            )
            reward += overlap_penalty
            log_penalty('overlap', abs(overlap_penalty))

            # 如果重叠时还几乎静止，额外惩罚
            if hunter.speed < 5.0:
                reward += self.stationary_penalty * 1.5
                log_penalty('stationary', abs(self.stationary_penalty * 1.5))

            # 调试输出
            if overlap_count >= 2 or max_overlap_ratio > 0.5:
                print(f"[重叠惩罚-HPO] 猎人 {hunter.id}: {overlap_count}个同类重叠, 程度={max_overlap_ratio:.2f}, 惩罚={overlap_penalty:.2f}")

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
        # 基础奖励参数 (强化: 大幅增强反应灵敏度)
        self.survival_bonus = 15.0  # 基础存活奖励
        self.escape_scale = 50.0  # 强化: 40.0 → 50.0 (增强25%, 更强逃跑动机)
        self.danger_distance = 400.0  # 扩大: 350.0 → 400.0 (增强14%, 更早感知)
        self.critical_distance = 200.0  # 扩大: 180.0 → 200.0 (增强11%, 更大危险区)
        self.critical_escape_multiplier = 5.0  # 强化: 4.0 → 5.0 (增强25%, 极度危险时更强反应)

        # 集群行为参数 (强化: 鼓励猎物形成集群)
        self.herd_scale = 5.0  # 强化: 3.0 → 5.0 (大幅加强集群奖励)
        self.herd_distance = 150.0  # 扩大: 120.0 → 150.0 (更大的集群感知范围)
        self.herd_cohesion_bonus = 25.0  # 增强: 15.0 → 25.0 (集群凝聚力奖励)
        self.herd_alignment_bonus = 20.0  # 增强: 10.0 → 20.0 (集群方向一致性奖励)
        self.evasion_bonus = 35.0  # 深度修复: 10.0 → 25.0 → 35.0 (提高250%)
        # 移动要求 (深度修复: 进一步降低速度要求)
        self.min_speed_threshold = 15.0  # 深度修复: 30.0 → 20.0 → 15.0 (进一步降低)
        self.movement_reward_scale = 12.0  # 深度修复: 5.0 → 8.0 → 12.0 (提高140%)
        # 静止惩罚 (深度修复: 进一步降低50%)
        self.stationary_penalty = -5.0  # 深度修复: -35.0 → -10.0 → -5.0 (降低86%)
        self.low_speed_penalty_scale = 1.0  # 深度修复: 8.0 → 2.0 → 1.0 (降低87.5%)
        self.death_penalty = -50.0
        self.flee_direction_scale = 5.0  # 降低: 10.0 → 5.0
        self.turn_reward_scale = 0.5  # 修改: 2.0 → 0.5 (防止原地转圈)
        self.escape_bonus_scale = 20.0  # 新增: 逃跑加成基数

        # 新增: 非意图性行为惩罚 (深度修复: 大幅降低)
        self.approach_hunter_penalty = -5.0  # 接近猎人惩罚 (深度修复: -20.0 → -5.0)
        self.facing_hunter_penalty = -3.0  # 面向猎人惩罚 (深度修复: -15.0 → -3.0)
        self.ineffective_escape_penalty = -2.0  # 无效逃跑惩罚 (深度修复: -8.0 → -2.0)

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

        # 持续逃跑时间追踪 (同步普通版本的改进)
        self.escape_streak = {}
        self.max_escape_multiplier = 5.0  # 提高: 3.0 → 5.0
        self.escape_buildup_steps = 8  # 降低: 10 → 8 (更快达到最大倍数)

        # 新增: 位置抖动检测 (防止多目标时原地停顿)
        self.position_history = {}  # prey_id -> [(x, y), ...]
        self.position_history_length = 5  # 追踪5步历史
        self.jitter_penalty = -3.0  # 原地抖动惩罚 (深度修复: -25.0 → -8.0 → -3.0, 降低88%)
        self.jitter_radius_threshold = 30.0  # 5步内活动半径<30px判定为抖动 (深度修复: 8.0 → 20.0 → 30.0, 进一步放宽)

        # 高角速度惩罚
        self.high_angular_velocity_penalty = -5.0
        self.angular_velocity_threshold = 0.3

        # 新增: 集群-逃跑冲突检测 (优化: 移除冲突惩罚，鼓励集群协同逃跑)
        self.herd_escape_conflict_penalty = 0.0  # 移除惩罚: -2.0 → 0.0 (允许集群灵活逃跑)
        self.dangerous_herd_distance = 200.0  # 猎人200px内不应聚集

        # 新增: 危险区域集群逃跑加成
        self.danger_herd_escape_bonus = 8.0  # 危险时集群协同逃跑的额外奖励
        self.critical_herd_escape_multiplier = 2.0  # 极度危险时的集群逃跑倍数

        # 新增: 多猎人威胁感知 (增强灵敏度)
        self.threat_decay_distance = 150.0  # 威胁权重衰减距离 (增强: 100.0 → 150.0, 更远感知)
        self.use_multi_hunter_threat = True  # 启用多猎人威胁感知

        # 同类重叠惩罚 (轻度惩罚过度重叠，但允许适当接近以形成集群)
        self.same_type_overlap_penalty = -1.0  # 基础惩罚 (进一步降低: -3.0 → -1.0, 鼓励集群)
        self.overlap_penalty_multiplier = 1.0   # 重叠程度倍数 (降低: 1.5 → 1.0)
        self.min_safe_distance_multiplier = 1.2 # 最小安全距离 = (r1+r2) * 1.2 (降低: 1.5 → 1.2, 允许更近距离)

    def detect_same_type_overlap(self, entity: EntityState, world: WorldState) -> Tuple[int, float]:
        """检测与同类的重叠情况

        Returns:
            overlap_count: 重叠的同类数量
            max_overlap_ratio: 最大重叠程度 [0, 1]
        """
        overlap_count = 0
        max_overlap_ratio = 0.0

        for other in world.entities:
            # 只检测同类，排除自己
            if other.type != entity.type or other.id == entity.id:
                continue

            # 计算距离
            dx = other.x - entity.x
            dy = other.y - entity.y
            distance = math.sqrt(dx**2 + dy**2)

            # 检测重叠 (距离 < 半径和)
            min_distance = entity.radius + other.radius
            if distance < min_distance:
                overlap_count += 1
                # 计算重叠程度 (归一化到0-1)
                overlap_depth = min_distance - distance
                overlap_ratio = overlap_depth / min_distance
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        return overlap_count, max_overlap_ratio

    def compute_threat_vector(self, prey: EntityState, curr_world: WorldState):
        """计算综合威胁向量 (同标准版)"""
        threat_x = 0.0
        threat_y = 0.0
        total_weight = 0.0
        visible_hunters = []
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

                if distance < self.danger_distance:
                    visible_hunters.append((entity, distance))
                    weight = math.exp(-distance / self.threat_decay_distance)
                    norm = distance + 1e-6
                    threat_x += (dx / norm) * weight
                    threat_y += (dy / norm) * weight
                    total_weight += weight

        if total_weight > 0:
            threat_x /= total_weight
            threat_y /= total_weight

        threat_angle = math.atan2(threat_y, threat_x)
        threat_magnitude = math.sqrt(threat_x**2 + threat_y**2)

        return threat_angle, threat_magnitude, visible_hunters, closest_hunter, min_distance

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

        # 1.5 位置抖动检测 (防止多目标时原地停顿)
        if prey.id not in self.position_history:
            self.position_history[prey.id] = []

        self.position_history[prey.id].append((prey.x, prey.y))
        if len(self.position_history[prey.id]) > self.position_history_length:
            self.position_history[prey.id].pop(0)

        if len(self.position_history[prey.id]) >= self.position_history_length:
            positions = self.position_history[prey.id]
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            max_radius = max(math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions)

            if max_radius < self.jitter_radius_threshold:
                reward += self.jitter_penalty
                log_penalty('jitter', abs(self.jitter_penalty))
                if prey.speed > 15.0:
                    print(f"[抖动惩罚-HPO] 猎物 {prey.id} 原地停顿! 半径={max_radius:.1f}px")

        # 2. 计算威胁（多猎人感知）
        if self.use_multi_hunter_threat:
            threat_angle, threat_magnitude, visible_hunters, closest_hunter, min_distance = \
                self.compute_threat_vector(prey, curr_world)

            if closest_hunter is None:
                if self.hpo_enhancer:
                    self.hpo_enhancer.update_outcome('escape')
                return reward
        else:
            # 传统单猎人逻辑
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
                if self.hpo_enhancer:
                    self.hpo_enhancer.update_outcome('escape')
                return reward

            dx = closest_hunter.x - prey.x
            dy = closest_hunter.y - prey.y
            threat_angle = math.atan2(dy, dx)
            threat_magnitude = min(1.0, self.danger_distance / (min_distance + 1.0))
            visible_hunters = [(closest_hunter, min_distance)]

        # 3. 即时逃跑奖励 (优化: 增强视觉威胁的即时反应)
        if min_distance < self.danger_distance:
            # 非线性危险等级: 距离越近反应越强烈
            danger_level = 1.0 - (min_distance / self.danger_distance)
            danger_level = danger_level ** 1.5  # 指数化，近距离威胁更紧急

            # 视觉范围内立即反应奖励
            if min_distance < self.danger_distance:
                immediate_response_bonus = 5.0 * (1.0 - min_distance / self.danger_distance)
                reward += immediate_response_bonus

            # 速度调制: 低速大幅削减奖励
            if prey.speed < self.min_speed_threshold:
                speed_mult = 0.15  # 提高: 0.1 → 0.15 (鼓励快速启动)
            else:
                speed_mult = min(prey.speed / 60.0, 1.0)  # 高速给满额

            escape_reward = weights['escape'] * danger_level * speed_mult
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

        # 5. 智能集群奖励 (优化: 跟随同类 + 优先逃避威胁)
        nearby_prey_list = []
        nearby_prey_count = 0

        # 收集附近的同类信息
        for entity in curr_world.entities:
            if entity.type == 'prey' and entity.id != prey.id:
                dx = entity.x - prey.x
                dy = entity.y - prey.y
                distance = math.sqrt(dx**2 + dy**2)

                # 计算最小安全距离
                min_safe_distance = (prey.radius + entity.radius) * self.min_safe_distance_multiplier

                # 只计入适当距离内的猎物 (不重叠)
                if distance < self.herd_distance and distance >= min_safe_distance:
                    nearby_prey_count += 1
                    nearby_prey_list.append((entity, distance, dx, dy))

        if nearby_prey_count > 0:
            # 基础集群奖励 (根据是否有威胁调整)
            in_danger = min_distance < self.danger_distance

            if not in_danger:
                # 安全区域: 鼓励聚集，形成防御集群
                herd_reward = self.herd_scale * math.sqrt(nearby_prey_count)
                reward += herd_reward

                # 5.1 集群凝聚力奖励: 与最近同类保持适当距离
                if nearby_prey_list:
                    closest_prey_dist = min(d for _, d, _, _ in nearby_prey_list)
                    # 理想距离: herd_distance的40-60%
                    ideal_dist_min = self.herd_distance * 0.4
                    ideal_dist_max = self.herd_distance * 0.6

                    if ideal_dist_min <= closest_prey_dist <= ideal_dist_max:
                        # 在理想距离范围内，给凝聚力奖励
                        cohesion_reward = self.herd_cohesion_bonus
                        reward += cohesion_reward

                # 5.2 集群方向一致性奖励: 与同类朝向相似
                if nearby_prey_list and len(nearby_prey_list) >= 2:
                    # 计算周围猎物的平均朝向
                    avg_angle_x = sum(math.cos(e.angle) for e, _, _, _ in nearby_prey_list)
                    avg_angle_y = sum(math.sin(e.angle) for e, _, _, _ in nearby_prey_list)
                    avg_angle = math.atan2(avg_angle_y, avg_angle_x)

                    # 计算当前猎物与平均朝向的对齐度
                    angle_diff_herd = prey.angle - avg_angle
                    while angle_diff_herd > math.pi:
                        angle_diff_herd -= 2 * math.pi
                    while angle_diff_herd < -math.pi:
                        angle_diff_herd += 2 * math.pi

                    alignment = math.cos(angle_diff_herd)
                    if alignment > 0.5:  # 朝向相似度超过50%
                        alignment_reward = self.herd_alignment_bonus * alignment
                        reward += alignment_reward

            else:
                # 危险区域: 优先逃避，强化集群协同逃跑
                danger_level = 1.0 - (min_distance / self.danger_distance)

                # 极度危险标记 (猎人在critical_distance内)
                is_critical_danger = min_distance < self.critical_distance

                # 检查是否在协同逃跑 (同类也在远离威胁)
                coordinated_escape_count = 0
                if nearby_prey_list and len(nearby_prey_list) >= 1:
                    # 检查周围猎物是否也在背离威胁
                    for entity, _, _, _ in nearby_prey_list:
                        # 计算该同类与威胁的角度差
                        entity_angle_diff = entity.angle - threat_angle
                        while entity_angle_diff > math.pi:
                            entity_angle_diff -= 2 * math.pi
                        while entity_angle_diff < -math.pi:
                            entity_angle_diff += 2 * math.pi
                        entity_flee_alignment = -math.cos(entity_angle_diff)

                        # 同类在背离威胁 (逃跑方向)
                        if entity_flee_alignment > 0.3:  # 至少30%对齐
                            coordinated_escape_count += 1

                    if coordinated_escape_count > 0:
                        # 集群协同逃跑: 给予强力奖励
                        # 基础协同奖励
                        coordinated_bonus = self.danger_herd_escape_bonus * math.sqrt(coordinated_escape_count)

                        # 极度危险时额外倍数
                        if is_critical_danger:
                            coordinated_bonus *= self.critical_herd_escape_multiplier

                        # 危险越高，协同奖励越高
                        coordinated_bonus *= (0.5 + 0.5 * danger_level)

                        reward += coordinated_bonus
                    else:
                        # 虽然有同类但未协同逃跑: 无惩罚，鼓励灵活应对
                        pass
                else:
                    # 独自在危险区: 正常逃跑，无额外惩罚
                    pass

        # 6. 逃跑方向奖励 (使用综合威胁角度)
        # threat_angle已由compute_threat_vector计算
        # 注意: threat_angle 指向猎人的方向 (猎物应该逃离的方向)

        # 计算猎物朝向与威胁方向的角度差
        angle_diff = prey.angle - threat_angle
        # 归一化到 [-π, π]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 计算逃跑对齐度: 猎物应该背离威胁 (angle_diff 接近 ±π 最好)
        # cos(angle_diff) = -1 表示完全背离 (最佳逃跑方向)
        # cos(angle_diff) = +1 表示面向威胁 (最差)
        flee_alignment = -math.cos(angle_diff)  # 反转后: +1=背离, -1=面向

        # 多猎人额外奖励
        multi_threat_multiplier = 1.0
        if self.use_multi_hunter_threat and len(visible_hunters) >= 2:
            multi_threat_multiplier = 1.0 + 0.2 * (len(visible_hunters) - 1)

        # 只在背离威胁时给奖励 (flee_alignment > 0)
        if flee_alignment > 0:
            flee_reward = self.flee_direction_scale * flee_alignment * multi_threat_multiplier
            reward += flee_reward

        # 7. 移动奖励与静止惩罚 (使用动态权重)
        speed_ratio = prey.speed / 52.0  # 更新: 45.0 → 52.0 (同步agent_config改进)

        if prey.speed < 2.0:
            reward += weights['stationary']
        elif prey.speed < self.min_speed_threshold:
            low_speed_penalty = -weights['low_speed'] * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:
            movement_reward = weights['movement'] * speed_ratio
            reward += movement_reward

        # 8. 增强逃跑加成 + 持续逃跑时间奖励 (强化版: 要求实质性逃离)
        prev_min_distance = self.prev_min_distances.get(prey.id)
        current_streak = self.escape_streak.get(prey.id, 0)

        # 判断是否在持续远离 (增加最小远离量要求)
        distance_change = 0.0
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance  # 正值表示远离
            # 要求每步至少远离0.5px才算有效逃跑
            if distance_change > 0.5:  # 实质性逃离
                current_streak = min(current_streak + 1, self.escape_buildup_steps)
            else:
                # 未有效逃离，快速衰减连击
                current_streak = max(0, current_streak - 2)  # 衰减更快

        self.escape_streak[prey.id] = current_streak

        # 计算逃跑倍数 (随持续时间线性增长)
        escape_multiplier = 1.0 + (self.max_escape_multiplier - 1.0) * (current_streak / self.escape_buildup_steps)

        # 极度危险时进一步提升倍数
        if min_distance < self.critical_distance:
            escape_multiplier *= self.critical_escape_multiplier

        # 强化逃跑加成: 高速 + 方向对齐 + 危险程度 + 持续时间 + 逃离速度
        if min_distance < self.danger_distance and prey.speed > self.min_speed_threshold:
            # 正确判断: 只在背离威胁时给奖励 (flee_alignment > 0.3)
            if flee_alignment > 0.3 and prev_min_distance is not None:
                # 计算逃离速度因子 (归一化到0-1)
                escape_speed_factor = 0.0
                if distance_change > 0.5:  # 有效逃离
                    escape_speed_factor = min(distance_change / 5.0, 1.0)
                else:
                    escape_speed_factor = 0.2  # 未有效逃离只给20%

                # 计算危险程度 (指数衰减)
                danger_level = 1.0 - (min_distance / self.danger_distance)
                # 极度危险时指数增长
                if min_distance < self.critical_distance:
                    danger_level = min(danger_level * 2.0, 1.0)

                # 基础逃跑奖励 (包含逃离速度)
                base_escape = self.escape_bonus_scale * speed_ratio * flee_alignment * danger_level * escape_speed_factor
                # 应用时间倍数
                escape_bonus = base_escape * escape_multiplier
                reward += escape_bonus

                # 调试输出
                if current_streak >= 3 and current_streak % 2 == 0:
                    critical_tag = "[极度危险]" if min_distance < self.critical_distance else ""
                    print(f"[HPO逃跑连击{critical_tag}] {prey.id}: {current_streak}步 × {escape_multiplier:.2f}倍, 逃离{distance_change:.1f}px, 距离{min_distance:.1f}px = +{escape_bonus:.2f}")
            else:
                # 非意图性逃跑惩罚: 在危险区域内但面向猎人 (flee_alignment < 0)
                if min_distance < self.danger_distance and flee_alignment < 0:
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    # 极度危险时加倍惩罚
                    if min_distance < self.critical_distance:
                        danger_level *= 2.0
                    # 惩罚强度与面向程度成正比 (abs(flee_alignment) 越大惩罚越重)
                    facing_penalty = self.facing_hunter_penalty * speed_ratio * danger_level * abs(flee_alignment)
                    reward += facing_penalty
                    log_penalty('facing_hunter', abs(facing_penalty))
                    if abs(facing_penalty) > 2.0:
                        print(f"[HPO面向惩罚] {prey.id}: 距离={min_distance:.1f}px, 面向程度={flee_alignment:.2f}, penalty={facing_penalty:.2f}")

        # 8.5 新增: 接近猎人惩罚 (强化版)
        if prev_min_distance is not None and closest_hunter is not None:
            distance_change = min_distance - prev_min_distance
            # 降低触发阈值: -2.0 → -1.0
            if distance_change < -1.0 and prey.speed > self.min_speed_threshold:
                # 高速接近猎人，重度惩罚
                if min_distance < self.danger_distance:
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    # 极度危险时3倍惩罚
                    if min_distance < self.critical_distance:
                        danger_level *= 3.0
                    approach_penalty = self.approach_hunter_penalty * min(abs(distance_change) / 10.0, 1.0) * danger_level
                    reward += approach_penalty
                    log_penalty('herd_conflict', abs(approach_penalty))
                    if abs(approach_penalty) > 3.0:
                        print(f"[HPO接近惩罚] {prey.id}: 接近{abs(distance_change):.1f}px, 距离{min_distance:.1f}px, penalty={approach_penalty:.2f}")

        # 8.6 新增: 无效逃跑惩罚 (危险区域内但不远离)
        if min_distance < self.danger_distance and prey.speed > 5.0:
            if current_streak == 0 and prev_min_distance is not None:
                distance_change = min_distance - prev_min_distance
                if abs(distance_change) < 1.0:  # 距离几乎没变化
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    # 极度危险时2倍惩罚
                    if min_distance < self.critical_distance:
                        danger_level *= 2.0
                    # 使用新的速度上限 52.0
                    ineffective_penalty = self.ineffective_escape_penalty * (prey.speed / 52.0) * danger_level
                    reward += ineffective_penalty

        # 9. 转向奖励 (使用动态权重)
        if closest_hunter is not None:
            prev_angle = self.prev_angles.get(prey.id)
            if prev_angle is not None:
                # 高角速度惩罚 (防止原地快速转圈)
                angle_change = prey.angle - prev_angle
                while angle_change > math.pi:
                    angle_change -= 2 * math.pi
                while angle_change < -math.pi:
                    angle_change += 2 * math.pi

                angular_velocity = abs(angle_change)
                if angular_velocity > self.angular_velocity_threshold:
                    high_angular_penalty = self.high_angular_velocity_penalty * (angular_velocity / math.pi)
                    reward += high_angular_penalty
                    log_penalty('high_angular', abs(high_angular_penalty))
                    if angular_velocity > 0.5:
                        print(f"[高角速度惩罚HPO Stage3] 猎物 {prey.id} 角速度={angular_velocity:.3f}rad, 惩罚={high_angular_penalty:.2f}")

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

        # 10. 同类重叠惩罚 (防止猎物过度聚集导致静止)
        overlap_count, max_overlap_ratio = self.detect_same_type_overlap(prey, curr_world)
        if overlap_count > 0:
            # 计算惩罚: 基础惩罚 × 重叠数量 × (1 + 重叠程度加成)
            overlap_penalty = (
                self.same_type_overlap_penalty *
                overlap_count *
                (1.0 + self.overlap_penalty_multiplier * max_overlap_ratio)
            )
            reward += overlap_penalty
            log_penalty('overlap', abs(overlap_penalty))

            # 如果重叠时还几乎静止，额外惩罚
            if prey.speed < 5.0:
                reward += self.stationary_penalty * 1.5
                log_penalty('stationary', abs(self.stationary_penalty * 1.5))

            # 调试输出
            if overlap_count >= 2 or max_overlap_ratio > 0.5:
                print(f"[重叠惩罚-HPO] 猎物 {prey.id}: {overlap_count}个同类重叠, 程度={max_overlap_ratio:.2f}, 惩罚={overlap_penalty:.2f}")

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
