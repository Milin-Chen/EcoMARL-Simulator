"""
分阶段训练的奖励函数
Curriculum Learning Reward Functions

每个阶段有不同的奖励设计，逐步提高难度
"""

import math
from typing import Dict, Tuple
from core.world import WorldState, EntityState
from config.agent_config import AgentConfig

# 导入训练日志器（可选，不影响功能）
try:
    from rl_env.training_logger import log_penalty, log_event
    _LOGGER_AVAILABLE = True
except ImportError:
    _LOGGER_AVAILABLE = False
    def log_penalty(*args, **kwargs): pass
    def log_event(*args, **kwargs): pass


class Stage1HunterReward:
    """阶段1: 猎人对静止猎物 - 超高捕食奖励 + 增强密集奖励 + 惩罚静止"""

    def __init__(self):
        # 加载配置
        self.agent_config = AgentConfig()
        # 超高捕食奖励
        self.predation_reward = 200.0

        # 接近奖励 (降低: 15.0 → 8.0, 防止低速刷分)
        self.approach_scale = 8.0
        self.max_approach_distance = 200.0

        # 减弱方向奖励 (15.0 → 8.0)
        self.direction_scale = 8.0

        # 移动奖励和静止惩罚 (强化: 防止原地停止)
        self.min_speed_threshold = 15.0  # 提高: 10.0 → 15.0 (要求更高速度)
        self.movement_reward_scale = 2.0  # 移动奖励系数
        self.stationary_penalty = -15.0  # 再次加强: -8.0 → -15.0 (必须移动!)
        self.low_speed_penalty_scale = 5.0  # 再次加强: 3.0 → 5.0 (严惩低速)

        # 大幅减弱转向奖励 (5.0 → 2.0 → 0.5) 防止原地转圈
        self.turn_reward_scale = 0.5  # 转向奖励系数

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

        # 新增: 目标锁定机制 (防止多目标摇摆)
        self.locked_target = {}  # hunter_id -> prey_id
        self.lock_duration = {}  # hunter_id -> remaining steps
        self.min_lock_steps = 8  # 最少锁定8步
        self.target_switch_penalty = -15.0  # 频繁切换目标的惩罚

        # 新增: 位置抖动检测 (防止原地打转) - 加强检测
        self.position_history = {}  # hunter_id -> [(x, y), ...]
        self.position_history_length = 5  # 追踪5步历史
        self.jitter_penalty = -20.0  # 原地抖动惩罚 (加强: -12.0 → -20.0)
        self.jitter_radius_threshold = 10.0  # 5步内活动半径<10px判定为抖动 (加强: 20.0 → 10.0)

        # 新增: 高角速度惩罚 (防止原地快速转向)
        self.high_angular_velocity_penalty = -5.0  # 高角速度惩罚
        self.angular_velocity_threshold = 0.3      # 角速度阈值

        # 新增: 视野丢失惩罚 (防止围绕猎物转圈但不在视野内)
        self.vision_loss_penalty = -10.0  # 猎物在附近但不在视野内的惩罚
        self.vision_check_distance = 200.0  # 检查视野的距离阈值
        self.fov_angle = self.agent_config.HUNTER_FOV_DEG * math.pi / 180.0 / 2.0  # 视野半角(弧度)

        # 新增: 同类重叠惩罚 (防止猎人聚集到同一点)
        self.same_type_overlap_penalty = -15.0  # 基础惩罚
        self.overlap_penalty_multiplier = 2.0   # 重叠程度倍数
        self.min_safe_distance_multiplier = 1.2 # 最小安全距离 = (r1+r2) * 1.2

    def detect_same_type_overlap(self, entity: EntityState, world: WorldState) -> Tuple[int, float]:
        """
        检测与同类的重叠情况

        Returns:
            overlap_count: 重叠的同类数量
            max_overlap_ratio: 最大重叠程度 [0, 1]
        """
        overlap_count = 0
        max_overlap_ratio = 0.0

        for other in world.entities:
            # 只检测同类
            if other.type != entity.type or other.id == entity.id:
                continue

            # 计算距离
            dx = other.x - entity.x
            dy = other.y - entity.y
            distance = math.sqrt(dx**2 + dy**2)

            # 检查重叠 (距离 < 两个半径之和)
            min_distance = entity.radius + other.radius
            if distance < min_distance:
                overlap_count += 1
                overlap_depth = min_distance - distance
                overlap_ratio = overlap_depth / min_distance
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        return overlap_count, max_overlap_ratio

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
            log_event('predation')  # 使用logger记录事件
            # 捕食成功，清除目标锁定
            if hunter.id in self.locked_target:
                del self.locked_target[hunter.id]
                del self.lock_duration[hunter.id]

        # 3. 位置抖动检测 (防止原地打转)
        if hunter.id not in self.position_history:
            self.position_history[hunter.id] = []

        # 更新位置历史
        self.position_history[hunter.id].append((hunter.x, hunter.y))
        if len(self.position_history[hunter.id]) > self.position_history_length:
            self.position_history[hunter.id].pop(0)

        # 检查是否原地抖动 (至少有5个历史位置)
        if len(self.position_history[hunter.id]) >= self.position_history_length:
            positions = self.position_history[hunter.id]
            # 计算活动半径 (位置的标准差)
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            max_radius = max(math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions)

            # 如果活动半径太小，判定为原地抖动
            if max_radius < self.jitter_radius_threshold:
                reward += self.jitter_penalty
                if hunter.speed > 10.0:  # 高速但原地抖动，额外惩罚
                    print(f"[抖动惩罚] 猎人 {hunter.id} 原地打转! 半径={max_radius:.1f}px, 惩罚={self.jitter_penalty:.1f}")

        # 4. 找到最近的猎物 (优先考虑锁定目标)
        closest_prey = None
        min_distance = float('inf')
        locked_prey = None

        # 收集所有猎物
        all_prey = [e for e in curr_world.entities if e.type == 'prey']

        if not all_prey:
            return reward  # 没有猎物了

        # 检查锁定目标是否还存在
        if hunter.id in self.locked_target:
            locked_prey_id = self.locked_target[hunter.id]
            for prey in all_prey:
                if prey.id == locked_prey_id:
                    locked_prey = prey
                    break

            # 如果锁定目标消失(被捕食)，清除锁定
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
            # 处于锁定期，必须追击锁定目标
            if locked_prey is not None:
                target_prey = locked_prey
                # 计算与锁定目标的距离
                dx = locked_prey.x - hunter.x
                dy = locked_prey.y - hunter.y
                min_distance = math.sqrt(dx**2 + dy**2)
                # 减少锁定计数
                self.lock_duration[hunter.id] -= 1
            else:
                # 锁定目标消失，使用最近目标
                target_prey = closest_prey
        else:
            # 未锁定或锁定期结束
            target_prey = closest_prey

            # 检查是否切换了目标
            if hunter.id in self.locked_target:
                prev_target_id = self.locked_target[hunter.id]
                if target_prey.id != prev_target_id:
                    # 目标切换，施加惩罚
                    reward += self.target_switch_penalty
                    print(f"[目标切换] 猎人 {hunter.id} 切换目标 {prev_target_id} -> {target_prey.id}, 惩罚={self.target_switch_penalty:.1f}")

            # 锁定新目标
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

            approach_reward = self.approach_scale * approach_factor * speed_mult
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

        # 5. 计算速度比率 (用于后续奖励)
        speed_ratio = hunter.speed / 50.0  # 归一化速度 (HUNTER_SPEED_MAX=50.0)

        # 5.5 方向对齐奖励 (修改: 要求必须移动才给奖励，防止原地转向刷分)
        dx = closest_prey.x - hunter.x
        dy = closest_prey.y - hunter.y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(hunter.angle - target_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        alignment = math.cos(angle_diff)

        # 要求: 必须有足够速度才给方向奖励 (防止原地转向)
        if alignment > 0.5 and hunter.speed > self.min_speed_threshold:
            # 方向奖励与速度挂钩
            direction_reward = self.direction_scale * alignment * speed_ratio
            reward += direction_reward

        # 5.6 视野丢失惩罚 (防止围绕猎物转圈但猎物不在视野内)
        if min_distance < self.vision_check_distance:
            # 检查猎物是否在视野范围内
            # 视野范围: 猎人朝向 ± fov_angle
            if angle_diff > self.fov_angle:
                # 猎物在附近但不在视野内，施加惩罚
                # 距离越近惩罚越大
                vision_loss_factor = 1.0 - (min_distance / self.vision_check_distance)
                vision_penalty = self.vision_loss_penalty * vision_loss_factor
                reward += vision_penalty
                log_penalty('vision_loss', abs(vision_penalty))

        # 6. 移动奖励与静止惩罚 (强化版)

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
                base_chase = self.chase_bonus_scale * speed_ratio * alignment * approach_factor
                # 应用时间倍数
                chase_bonus = base_chase * chase_multiplier
                reward += chase_bonus

                # 调试输出 (显示接近量)
                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[追击连击] {hunter.id}: {current_streak}步 × {chase_multiplier:.2f}倍, 接近{distance_change:.1f}px = +{chase_bonus:.2f}")
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

                # 高角速度惩罚 (防止原地快速转圈)
                angular_velocity = abs(angle_change)
                if angular_velocity > self.angular_velocity_threshold:
                    # 超过阈值，施加惩罚
                    high_angular_penalty = self.high_angular_velocity_penalty * (angular_velocity / math.pi)
                    reward += high_angular_penalty
                    log_penalty('high_angular', abs(high_angular_penalty))

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

        # 9. 同类重叠惩罚 (防止猎人聚集到同一点)
        overlap_count, max_overlap_ratio = self.detect_same_type_overlap(hunter, curr_world)
        if overlap_count > 0:
            # 基础惩罚 × 重叠数量 × (1 + 重叠程度倍数 × 重叠比例)
            overlap_penalty = (
                self.same_type_overlap_penalty *
                overlap_count *
                (1.0 + self.overlap_penalty_multiplier * max_overlap_ratio)
            )
            reward += overlap_penalty

            # 重叠时静止额外惩罚 (重叠且低速移动更严重)
            if hunter.speed < 5.0:
                reward += self.stationary_penalty * 1.5

            # 记录到统计
            log_penalty('overlap', abs(overlap_penalty))

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

        # 接近奖励 (降低: 12.0 → 6.0, 防止低速刷分)
        self.approach_scale = 6.0
        self.max_approach_distance = 200.0

        # 减弱方向奖励 (12.0 → 6.0)
        self.direction_scale = 6.0

        # 协作奖励
        self.cooperation_scale = 2.0
        self.cooperation_distance = 150.0

        # 移动奖励和静止惩罚 (强化: 防止原地停止)
        self.min_speed_threshold = 15.0  # 提高: 10.0 → 15.0 (要求更高速度)
        self.movement_reward_scale = 2.0
        self.stationary_penalty = -15.0  # 再次加强: -8.0 → -15.0 (必须移动!)
        self.low_speed_penalty_scale = 5.0  # 再次加强: 3.0 → 5.0 (严惩低速)

        # 减弱转向奖励 (5.0 → 2.0 → 0.5 防止原地转圈)
        self.turn_reward_scale = 0.5

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

        # 高角速度惩罚
        self.high_angular_velocity_penalty = -5.0
        self.angular_velocity_threshold = 0.3

        # 视野丢失惩罚 (防止围绕猎物转圈但不在视野内)
        self.vision_loss_penalty = -10.0
        self.vision_check_distance = 200.0
        self.fov_angle = self.agent_config.HUNTER_FOV_DEG * math.pi / 180.0 / 2.0

        # 同类重叠惩罚 (Stage2协作训练中更严格)
        self.same_type_overlap_penalty = -18.0  # 基础惩罚 (Stage2更严格)
        self.overlap_penalty_multiplier = 2.5   # 重叠程度倍数
        self.min_safe_distance_multiplier = 1.3 # 最小安全距离

    def detect_same_type_overlap(self, entity: EntityState, world: WorldState) -> Tuple[int, float]:
        """检测与同类的重叠情况 (同Stage1)"""
        overlap_count = 0
        max_overlap_ratio = 0.0

        for other in world.entities:
            if other.type != entity.type or other.id == entity.id:
                continue

            dx = other.x - entity.x
            dy = other.y - entity.y
            distance = math.sqrt(dx**2 + dy**2)

            min_distance = entity.radius + other.radius
            if distance < min_distance:
                overlap_count += 1
                overlap_depth = min_distance - distance
                overlap_ratio = overlap_depth / min_distance
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        return overlap_count, max_overlap_ratio

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
            # 捕食成功，清除目标锁定
            if hunter.id in self.locked_target:
                del self.locked_target[hunter.id]
                del self.lock_duration[hunter.id]

        # 3. 位置抖动检测 (防止原地打转)
        if hunter.id not in self.position_history:
            self.position_history[hunter.id] = []

        # 更新位置历史
        self.position_history[hunter.id].append((hunter.x, hunter.y))
        if len(self.position_history[hunter.id]) > self.position_history_length:
            self.position_history[hunter.id].pop(0)

        # 检查是否原地抖动 (至少有5个历史位置)
        if len(self.position_history[hunter.id]) >= self.position_history_length:
            positions = self.position_history[hunter.id]
            # 计算活动半径
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            max_radius = max(math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions)

            # 如果活动半径太小，判定为原地抖动
            if max_radius < self.jitter_radius_threshold:
                reward += self.jitter_penalty
                if hunter.speed > 10.0:  # 高速但原地抖动，额外惩罚
                    print(f"[抖动惩罚] 猎人 {hunter.id} 原地打转! 半径={max_radius:.1f}px, 惩罚={self.jitter_penalty:.1f}")

        # 4. 找到最近的猎物 (优先考虑锁定目标)
        closest_prey = None
        min_distance = float('inf')
        locked_prey = None

        # 收集所有猎物
        all_prey = [e for e in curr_world.entities if e.type == 'prey']

        if not all_prey:
            return reward  # 没有猎物了

        # 检查锁定目标是否还存在
        if hunter.id in self.locked_target:
            locked_prey_id = self.locked_target[hunter.id]
            for prey in all_prey:
                if prey.id == locked_prey_id:
                    locked_prey = prey
                    break

            # 如果锁定目标消失(被捕食)，清除锁定
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
            # 处于锁定期，必须追击锁定目标
            if locked_prey is not None:
                target_prey = locked_prey
                # 计算与锁定目标的距离
                dx = locked_prey.x - hunter.x
                dy = locked_prey.y - hunter.y
                min_distance = math.sqrt(dx**2 + dy**2)
                # 减少锁定计数
                self.lock_duration[hunter.id] -= 1
            else:
                # 锁定目标消失，使用最近目标
                target_prey = closest_prey
        else:
            # 未锁定或锁定期结束
            target_prey = closest_prey

            # 检查是否切换了目标
            if hunter.id in self.locked_target:
                prev_target_id = self.locked_target[hunter.id]
                if target_prey.id != prev_target_id:
                    # 目标切换，施加惩罚
                    reward += self.target_switch_penalty
                    print(f"[目标切换] 猎人 {hunter.id} 切换目标 {prev_target_id} -> {target_prey.id}, 惩罚={self.target_switch_penalty:.1f}")

            # 锁定新目标
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

            approach_reward = self.approach_scale * approach_factor * speed_mult
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

        # 5.5 视野丢失惩罚 (防止围绕猎物转圈但猎物不在视野内)
        if min_distance < self.vision_check_distance:
            if angle_diff > self.fov_angle:
                vision_loss_factor = 1.0 - (min_distance / self.vision_check_distance)
                vision_penalty = self.vision_loss_penalty * vision_loss_factor
                reward += vision_penalty
                if abs(vision_penalty) > 5.0:
                    print(f"[视野丢失Stage2] 猎人 {hunter.id} 猎物在{min_distance:.1f}px但不在视野, 惩罚={vision_penalty:.2f}")

        # 6. 协作奖励 (新增!) - 奖励多个猎人围攻同一猎物 (但要保持安全距离)
        other_hunters_nearby = 0
        for entity in curr_world.entities:
            if entity.type == 'hunter' and entity.id != hunter.id:
                # 检查这个猎人是否也在追同一个猎物
                dx_other = closest_prey.x - entity.x
                dy_other = closest_prey.y - entity.y
                distance_other = math.sqrt(dx_other**2 + dy_other**2)

                # 同时检查与当前猎人的距离，避免过度靠近
                dx_hunter = entity.x - hunter.x
                dy_hunter = entity.y - hunter.y
                distance_to_hunter = math.sqrt(dx_hunter**2 + dy_hunter**2)
                min_safe_distance = (hunter.radius + entity.radius) * self.min_safe_distance_multiplier

                # 只有在适当距离内（不重叠）才计入协作
                if distance_other < self.cooperation_distance and distance_to_hunter >= min_safe_distance:
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

        # 8. 增强追击加成 + 持续追击时间奖励 (强化版: 要求实质性接近)
        prev_distance = self.prev_distances.get(hunter.id)
        current_streak = self.chase_streak.get(hunter.id, 0)

        # 判断是否在持续接近 (增加最小接近量要求)
        distance_change = 0.0
        if prev_distance is not None and closest_prey is not None:
            distance_change = prev_distance - min_distance
            # 要求每步至少接近0.5px才算有效追击
            if distance_change > 0.5:  # 实质性接近
                current_streak = min(current_streak + 1, self.chase_buildup_steps)
                # 距离进度奖励 (强化)
                progress_reward = 10.0 * min(distance_change / 10.0, 1.0)
                reward += progress_reward
            else:
                # 未有效接近，快速衰减连击
                current_streak = max(0, current_streak - 2)  # 衰减更快

        self.chase_streak[hunter.id] = current_streak

        if closest_prey is not None:
            self.prev_distances[hunter.id] = min_distance

        # 计算追击倍数
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
                base_chase = self.chase_bonus_scale * speed_ratio * alignment * approach_factor
                # 应用时间倍数
                chase_bonus = base_chase * chase_multiplier
                reward += chase_bonus

                # 调试输出 (显示接近量)
                if current_streak >= 3 and current_streak % 2 == 0:
                    print(f"[追击连击Stage2] {hunter.id}: {current_streak}步 × {chase_multiplier:.2f}倍, 接近{distance_change:.1f}px = +{chase_bonus:.2f}")
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
                # 高角速度惩罚 (防止原地快速转圈)
                angle_change = hunter.angle - prev_angle
                # 归一化到 [-π, π]
                while angle_change > math.pi:
                    angle_change -= 2 * math.pi
                while angle_change < -math.pi:
                    angle_change += 2 * math.pi

                angular_velocity = abs(angle_change)
                if angular_velocity > self.angular_velocity_threshold:
                    high_angular_penalty = self.high_angular_velocity_penalty * (angular_velocity / math.pi)
                    reward += high_angular_penalty
                    if angular_velocity > 0.5:
                        print(f"[高角速度惩罚Stage2] 猎人 {hunter.id} 角速度={angular_velocity:.3f}rad, 惩罚={high_angular_penalty:.2f}")

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

        # 10. 同类重叠惩罚 (Stage2协作训练中更重要)
        overlap_count, max_overlap_ratio = self.detect_same_type_overlap(hunter, curr_world)
        if overlap_count > 0:
            overlap_penalty = (
                self.same_type_overlap_penalty *
                overlap_count *
                (1.0 + self.overlap_penalty_multiplier * max_overlap_ratio)
            )
            reward += overlap_penalty

            # 重叠时静止额外惩罚
            if hunter.speed < 5.0:
                reward += self.stationary_penalty * 1.5

            # 调试输出
            if abs(overlap_penalty) > 10.0:
                print(f"[重叠惩罚Stage2] 猎人 {hunter.id}: {overlap_count}个同类重叠, "
                      f"程度={max_overlap_ratio:.2f}, 惩罚={overlap_penalty:.2f}")

        self.prev_positions[hunter.id] = (hunter.x, hunter.y)
        self.prev_angles[hunter.id] = hunter.angle

        return reward


class Stage3PreyReward:
    """阶段3: 猎物训练 - 强逃跑与集群奖励 + 惩罚静止"""

    def __init__(self):
        # 加载配置
        self.agent_config = AgentConfig()

        # 存活奖励 (大幅提高: 0.5 → 5.0，激励生存本能)
        self.survival_bonus = 5.0

        # 逃跑奖励 (降低: 30.0 → 15.0, 防止静止刷分)
        self.escape_scale = 15.0
        # 扩大危险感知距离 (200.0 → 300.0，更早发现威胁)
        self.danger_distance = 300.0
        # 新增: 极度危险距离 (150px内3倍加成)
        self.critical_distance = 150.0
        self.critical_escape_multiplier = 3.0

        # 削弱集群奖励 (3.0 → 1.0，避免聚堆)
        self.herd_scale = 1.0
        self.herd_distance = 100.0

        # 大幅增强躲避奖励 (10.0 → 25.0)
        self.evasion_bonus = 25.0

        # 提高移动要求 (强化: 猎物必须快速逃跑)
        self.min_speed_threshold = 30.0  # 大幅提高: 20.0 → 30.0 (要求高速逃跑)
        self.movement_reward_scale = 5.0  # 大幅提高: 3.0 → 5.0 (奖励高速移动)
        # 极大加强静止惩罚 (严厉惩罚原地不动)
        self.stationary_penalty = -35.0  # 再次加强: -20.0 → -35.0 (必须逃跑!)
        self.low_speed_penalty_scale = 8.0  # 再次加强: 5.0 → 8.0 (严惩低速)

        # 被捕食惩罚
        self.death_penalty = -50.0

        # 保持逃跑方向奖励
        self.flee_direction_scale = 5.0

        # 保持转向奖励
        self.turn_reward_scale = 0.5  # 修改: 2.0 → 0.5 (防止原地转圈)

        # 增强逃跑加成 (12.0 → 20.0)
        self.escape_bonus_scale = 20.0

        self.prev_positions = {}
        self.prev_min_distances = {}
        self.prev_angles = {}

        # 持续逃跑时间追踪 (更激进)
        self.escape_streak = {}
        self.max_escape_multiplier = 5.0  # 提高: 3.0 → 5.0
        self.escape_buildup_steps = 8  # 降低: 10 → 8 (更快达到最大倍数)

        # 大幅加强非意图性行为惩罚
        self.approach_hunter_penalty = -20.0  # 提高: -8.0 → -20.0
        self.facing_hunter_penalty = -15.0  # 提高: -4.0 → -15.0
        self.ineffective_escape_penalty = -8.0  # 提高: -3.0 → -8.0

        # 新增: 位置抖动检测 (防止多目标时原地停顿)
        self.position_history = {}  # prey_id -> [(x, y), ...]
        self.position_history_length = 5  # 追踪5步历史
        self.jitter_penalty = -25.0  # 原地抖动惩罚 (修改: -15.0 → -25.0 加强惩罚)
        self.jitter_radius_threshold = 8.0  # 5步内活动半径<8px判定为抖动 (修改: 15.0 → 8.0 收紧阈值)

        # 高角速度惩罚
        self.high_angular_velocity_penalty = -5.0
        self.angular_velocity_threshold = 0.3

        # 新增: 集群-逃跑冲突检测
        self.herd_escape_conflict_penalty = -10.0  # 在危险时聚集的惩罚
        self.dangerous_herd_distance = 200.0  # 猎人200px内不应聚集

        # 新增: 多猎人威胁感知
        self.threat_decay_distance = 100.0  # 威胁权重衰减距离
        self.use_multi_hunter_threat = True  # 启用多猎人威胁感知

        # 同类重叠惩罚 (猎物最严格，容易恐慌聚集)
        self.same_type_overlap_penalty = -20.0  # 基础惩罚 (最严格)
        self.overlap_penalty_multiplier = 3.0   # 重叠程度倍数
        self.min_safe_distance_multiplier = 1.5 # 最小安全距离 (猎物需要更大空间)

    def detect_same_type_overlap(self, entity: EntityState, world: WorldState) -> Tuple[int, float]:
        """检测与同类的重叠情况 (同Stage1/Stage2)"""
        overlap_count = 0
        max_overlap_ratio = 0.0

        for other in world.entities:
            if other.type != entity.type or other.id == entity.id:
                continue

            dx = other.x - entity.x
            dy = other.y - entity.y
            distance = math.sqrt(dx**2 + dy**2)

            min_distance = entity.radius + other.radius
            if distance < min_distance:
                overlap_count += 1
                overlap_depth = min_distance - distance
                overlap_ratio = overlap_depth / min_distance
                max_overlap_ratio = max(max_overlap_ratio, overlap_ratio)

        return overlap_count, max_overlap_ratio

    def compute_threat_vector(self, prey: EntityState, curr_world: WorldState):
        """计算综合威胁向量

        考虑所有可见猎人的位置，计算加权平均威胁方向
        越近的猎人权重越大（指数衰减）

        Returns:
            threat_angle: 综合威胁角度
            threat_magnitude: 威胁强度 (0-1)
            visible_hunters: 可见猎人列表 [(hunter, distance), ...]
            closest_hunter: 最近的猎人 (兼容性)
            min_distance: 最近距离 (兼容性)
        """
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

                # 记录最近猎人（兼容性）
                if distance < min_distance:
                    min_distance = distance
                    closest_hunter = entity

                # 只考虑危险范围内的猎人
                if distance < self.danger_distance:
                    visible_hunters.append((entity, distance))

                    # 威胁权重: 指数衰减，越近权重越大
                    weight = math.exp(-distance / self.threat_decay_distance)

                    # 归一化方向向量
                    norm = distance + 1e-6
                    threat_x += (dx / norm) * weight
                    threat_y += (dy / norm) * weight
                    total_weight += weight

        # 计算综合威胁方向
        if total_weight > 0:
            threat_x /= total_weight
            threat_y /= total_weight

        threat_angle = math.atan2(threat_y, threat_x)
        threat_magnitude = math.sqrt(threat_x**2 + threat_y**2)

        return threat_angle, threat_magnitude, visible_hunters, closest_hunter, min_distance

    def compute_reward(self, prey: EntityState, prev_prey: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算阶段3猎物奖励"""
        reward = 0.0

        # 1. 基础存活奖励
        reward += self.survival_bonus

        # 1.5 位置抖动检测 (防止多目标时原地停顿)
        if prey.id not in self.position_history:
            self.position_history[prey.id] = []

        # 更新位置历史
        self.position_history[prey.id].append((prey.x, prey.y))
        if len(self.position_history[prey.id]) > self.position_history_length:
            self.position_history[prey.id].pop(0)

        # 检查是否原地抖动 (至少有5个历史位置)
        if len(self.position_history[prey.id]) >= self.position_history_length:
            positions = self.position_history[prey.id]
            # 计算活动半径
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            max_radius = max(math.sqrt((x - center_x)**2 + (y - center_y)**2) for x, y in positions)

            # 如果活动半径太小，判定为原地抖动
            if max_radius < self.jitter_radius_threshold:
                reward += self.jitter_penalty
                if prey.speed > 15.0:  # 中高速但原地抖动，额外严重
                    print(f"[抖动惩罚] 猎物 {prey.id} 原地停顿! 半径={max_radius:.1f}px, 惩罚={self.jitter_penalty:.1f}")

        # 2. 计算威胁（多猎人感知 or 单猎人）
        if self.use_multi_hunter_threat:
            # 使用多猎人威胁综合感知
            threat_angle, threat_magnitude, visible_hunters, closest_hunter, min_distance = \
                self.compute_threat_vector(prey, curr_world)

            if closest_hunter is None:
                return reward  # 没有猎人了

            # 调试输出
            if len(visible_hunters) > 1 and prey.id == 'prey_0':  # 只输出第一个猎物的信息
                print(f"[多猎人威胁] {prey.id}: {len(visible_hunters)}个猎人, 综合威胁={threat_magnitude:.2f}, 最近={min_distance:.1f}px")
        else:
            # 传统单猎人逻辑（保留用于对比）
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
                return reward

            # 单猎人威胁
            dx = closest_hunter.x - prey.x
            dy = closest_hunter.y - prey.y
            threat_angle = math.atan2(dy, dx)
            threat_magnitude = min(1.0, self.danger_distance / (min_distance + 1.0))
            visible_hunters = [(closest_hunter, min_distance)]

        # 3. 逃跑奖励 (必须配合速度! 防止静止刷分)
        if min_distance < self.danger_distance:
            danger_level = 1.0 - (min_distance / self.danger_distance)

            # 速度调制: 低速大幅削减奖励
            if prey.speed < self.min_speed_threshold:
                speed_mult = 0.1  # 低速只给10%
            else:
                speed_mult = min(prey.speed / 60.0, 1.0)  # 高速给满额

            escape_reward = self.escape_scale * danger_level * speed_mult
            reward += escape_reward

        # 4. 躲避奖励 (距离增加)
        prev_min_distance = self.prev_min_distances.get(prey.id)
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance
            if distance_change > 0:  # 拉开距离
                evasion_reward = self.evasion_bonus * (distance_change / 10.0)
                reward += min(evasion_reward, 2.0)  # 限制上限

        # 5. 集群奖励 (修改: 只在安全时奖励聚集，且保持安全距离)
        nearby_prey = 0
        for entity in curr_world.entities:
            if entity.type == 'prey' and entity.id != prey.id:
                dx = entity.x - prey.x
                dy = entity.y - prey.y
                distance = math.sqrt(dx**2 + dy**2)

                # 计算最小安全距离
                min_safe_distance = (prey.radius + entity.radius) * self.min_safe_distance_multiplier

                # 只计入适当距离内的猎物 (不重叠)
                if distance < self.herd_distance and distance >= min_safe_distance:
                    nearby_prey += 1

        if nearby_prey > 0:
            # 只在相对安全时才奖励聚集
            if min_distance > self.dangerous_herd_distance:
                # 安全区域，鼓励聚集
                herd_reward = self.herd_scale * math.sqrt(nearby_prey)
                reward += herd_reward
            else:
                # 危险区域聚集，强烈惩罚 (应该逃跑而不是聚集)
                herd_conflict_penalty = self.herd_escape_conflict_penalty * nearby_prey * (1.0 - min_distance / self.dangerous_herd_distance)
                reward += herd_conflict_penalty
                if abs(herd_conflict_penalty) > 5.0:
                    print(f"[聚集冲突] 猎物 {prey.id} 危险时聚集! 猎人距离={min_distance:.1f}px, 惩罚={herd_conflict_penalty:.2f}")

        # 6. 增强逃跑方向奖励 (背离综合威胁)
        # 使用compute_threat_vector计算的threat_angle

        # 计算逃跑方向对齐度
        angle_diff = abs(prey.angle - threat_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # 计算逃跑对齐度 (背离威胁的程度)
        flee_alignment = math.cos(angle_diff + math.pi)  # 翻转cos

        # 多猎人额外奖励
        multi_threat_multiplier = 1.0
        if self.use_multi_hunter_threat and len(visible_hunters) >= 2:
            # 成功应对多个猎人，额外倍数
            multi_threat_multiplier = 1.0 + 0.2 * (len(visible_hunters) - 1)  # 每多1个猎人+20%

        # 7. 计算速度比率 (用于后续奖励)
        speed_ratio = prey.speed / 60.0  # 归一化速度 (PREY_SPEED_MAX=60.0)

        # 奖励背离威胁的方向 (修改: 必须高速逃跑才给奖励，防止原地转向)
        if angle_diff > math.pi / 2 and prey.speed > self.min_speed_threshold:
            if flee_alignment > 0:
                # 逃跑方向奖励与速度挂钩 (速度越快奖励越高)
                flee_reward = self.flee_direction_scale * flee_alignment * speed_ratio * multi_threat_multiplier
                reward += flee_reward

                # 调试输出（多猎人情况）
                if len(visible_hunters) >= 2 and flee_reward > 3.0:
                    print(f"[多猎人逃跑] {prey.id}: {len(visible_hunters)}个猎人, 方向奖励={flee_reward:.2f} (倍数={multi_threat_multiplier:.2f})")

        # 7.5 移动奖励与静止惩罚 (强制猎物移动)

        if prey.speed < 2.0:  # 静止
            reward += self.stationary_penalty
        elif prey.speed < self.min_speed_threshold:  # 低速
            low_speed_penalty = -self.low_speed_penalty_scale * (1.0 - speed_ratio)
            reward += low_speed_penalty
        else:  # 高速
            movement_reward = self.movement_reward_scale * speed_ratio
            reward += movement_reward

        # 8. 增强逃跑加成 + 持续逃跑时间奖励 (强化版: 要求实质性逃离)
        prev_min_distance = self.prev_min_distances.get(prey.id)
        current_streak = self.escape_streak.get(prey.id, 0)

        # 判断是否在持续远离 (增加最小远离量要求)
        distance_change = 0.0
        if prev_min_distance is not None:
            distance_change = min_distance - prev_min_distance
            # 要求每步至少远离0.5px才算有效逃跑
            if distance_change > 0.5:  # 实质性逃离
                current_streak = min(current_streak + 1, self.escape_buildup_steps)
            else:
                # 未有效逃离，快速衰减连击
                current_streak = max(0, current_streak - 2)  # 衰减更快

        self.escape_streak[prey.id] = current_streak

        # 计算逃跑倍数
        escape_multiplier = 1.0 + (self.max_escape_multiplier - 1.0) * (current_streak / self.escape_buildup_steps)

        # 极度危险时进一步提升倍数
        if min_distance < self.critical_distance:
            escape_multiplier *= self.critical_escape_multiplier

        # 强化逃跑加成: 在危险时高速逃跑 + 持续加速逃离
        if min_distance < self.danger_distance and prey.speed > self.min_speed_threshold:
            if angle_diff > math.pi / 2 and prev_min_distance is not None:  # 正在逃跑
                # 计算逃离速度因子 (归一化到0-1，逃离越快奖励越高)
                escape_speed_factor = 0.0
                if distance_change > 0.5:  # 有效逃离
                    # 远离5px以上得到满分，线性插值
                    escape_speed_factor = min(distance_change / 5.0, 1.0)
                else:
                    # 未有效逃离，即使方向正确也只给少量奖励
                    escape_speed_factor = 0.2

                # 计算危险程度 (指数衰减，越近越危险)
                danger_level = 1.0 - (min_distance / self.danger_distance)
                # 极度危险时指数增长
                if min_distance < self.critical_distance:
                    danger_level = min(danger_level * 2.0, 1.0)

                # 基础逃跑奖励 (包含逃离速度)
                base_escape = self.escape_bonus_scale * speed_ratio * flee_alignment * danger_level * escape_speed_factor
                # 应用时间倍数
                escape_bonus = base_escape * escape_multiplier
                reward += escape_bonus

                # 调试输出 (显示逃离量)
                if current_streak >= 3 and current_streak % 2 == 0:
                    critical_tag = "[极度危险]" if min_distance < self.critical_distance else ""
                    print(f"[逃跑连击{critical_tag}] {prey.id}: {current_streak}步 × {escape_multiplier:.2f}倍, 逃离{distance_change:.1f}px, 距离{min_distance:.1f}px = +{escape_bonus:.2f}")
            else:
                # 非意图性逃跑惩罚: 在危险区域内但朝向猎人
                if min_distance < self.danger_distance:
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    # 极度危险时加倍惩罚
                    if min_distance < self.critical_distance:
                        danger_level *= 2.0
                    facing_penalty = self.facing_hunter_penalty * speed_ratio * danger_level
                    reward += facing_penalty
                    if abs(facing_penalty) > 2.0:
                        print(f"[面向惩罚] {prey.id}: 距离={min_distance:.1f}, penalty={facing_penalty:.2f}")

        # 8.5 新增: 接近猎人惩罚 (强化版)
        if prev_min_distance is not None and closest_hunter is not None:
            distance_change = min_distance - prev_min_distance
            # 降低触发阈值: -2.0 → -1.0，更严格
            if distance_change < -1.0 and prey.speed > self.min_speed_threshold:
                # 高速接近猎人，重度惩罚
                if min_distance < self.danger_distance:
                    danger_level = 1.0 - (min_distance / self.danger_distance)
                    # 极度危险时3倍惩罚
                    if min_distance < self.critical_distance:
                        danger_level *= 3.0
                    approach_penalty = self.approach_hunter_penalty * min(abs(distance_change) / 10.0, 1.0) * danger_level
                    reward += approach_penalty
                    if abs(approach_penalty) > 3.0:
                        print(f"[接近惩罚] {prey.id}: 接近{abs(distance_change):.1f}px, 距离{min_distance:.1f}px, penalty={approach_penalty:.2f}")

        # 8.6 新增: 无效逃跑惩罚 (危险区域内但不远离) - 强化版
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

        # 9. 减弱转向奖励 (奖励背离猎人转向)
        if closest_hunter is not None:
            prev_angle = self.prev_angles.get(prey.id)
            if prev_angle is not None:
                # 高角速度惩罚 (防止原地快速转圈)
                angle_change = prey.angle - prev_angle
                # 归一化到 [-π, π]
                while angle_change > math.pi:
                    angle_change -= 2 * math.pi
                while angle_change < -math.pi:
                    angle_change += 2 * math.pi

                angular_velocity = abs(angle_change)
                if angular_velocity > self.angular_velocity_threshold:
                    high_angular_penalty = self.high_angular_velocity_penalty * (angular_velocity / math.pi)
                    reward += high_angular_penalty
                    if angular_velocity > 0.5:
                        print(f"[高角速度惩罚Stage3] 猎物 {prey.id} 角速度={angular_velocity:.3f}rad, 惩罚={high_angular_penalty:.2f}")

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

        # 10. 同类重叠惩罚 (猎物最严格，防止恐慌聚集)
        overlap_count, max_overlap_ratio = self.detect_same_type_overlap(prey, curr_world)
        if overlap_count > 0:
            overlap_penalty = (
                self.same_type_overlap_penalty *
                overlap_count *
                (1.0 + self.overlap_penalty_multiplier * max_overlap_ratio)
            )
            reward += overlap_penalty

            # 重叠时静止额外惩罚 (猎物重叠且低速更严重)
            if prey.speed < 10.0:  # 猎物应该更快移动
                reward += self.stationary_penalty * 2.0

            # 调试输出
            if abs(overlap_penalty) > 15.0:
                print(f"[重叠惩罚Stage3] 猎物 {prey.id}: {overlap_count}个同类重叠, "
                      f"程度={max_overlap_ratio:.2f}, 惩罚={overlap_penalty:.2f}")

        # 更新记录
        self.prev_positions[prey.id] = (prey.x, prey.y)
        self.prev_min_distances[prey.id] = min_distance
        self.prev_angles[prey.id] = prey.angle

        return reward


class Stage4JointReward:
    """阶段4: 联合微调 - 使用修复后的Stage2猎人和Stage3猎物奖励"""

    def __init__(self):
        # 使用修复前的标准课程奖励 (将被HPO替换)
        self.hunter_reward = Stage2HunterReward()
        self.prey_reward = Stage3PreyReward()

    def compute_reward(self, entity: EntityState, prev_entity: EntityState,
                      prev_world: WorldState, curr_world: WorldState) -> float:
        """计算阶段4奖励 - 根据角色使用不同的奖励函数"""
        if entity.type == "hunter":
            return self.hunter_reward.compute_reward(entity, prev_entity, prev_world, curr_world)
        else:  # prey
            return self.prey_reward.compute_reward(entity, prev_entity, prev_world, curr_world)


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
            # Stage4使用joint_reward，需要重置其子奖励函数
            joint = self.joint_reward

            # 重置hunter_reward
            if hasattr(joint, 'hunter_reward') and joint.hunter_reward is not None:
                if hasattr(joint.hunter_reward, 'prev_positions'):
                    joint.hunter_reward.prev_positions = {}
                if hasattr(joint.hunter_reward, 'prev_distances'):
                    joint.hunter_reward.prev_distances = {}
                if hasattr(joint.hunter_reward, 'prev_angles'):
                    joint.hunter_reward.prev_angles = {}
                if hasattr(joint.hunter_reward, 'prev_speeds'):
                    joint.hunter_reward.prev_speeds = {}
                if hasattr(joint.hunter_reward, 'chase_streak'):
                    joint.hunter_reward.chase_streak = {}
                if hasattr(joint.hunter_reward, 'position_history'):
                    joint.hunter_reward.position_history = {}

            # 重置prey_reward
            if hasattr(joint, 'prey_reward') and joint.prey_reward is not None:
                if hasattr(joint.prey_reward, 'prev_positions'):
                    joint.prey_reward.prev_positions = {}
                if hasattr(joint.prey_reward, 'prev_min_distances'):
                    joint.prey_reward.prev_min_distances = {}
                if hasattr(joint.prey_reward, 'prev_angles'):
                    joint.prey_reward.prev_angles = {}
                if hasattr(joint.prey_reward, 'escape_streak'):
                    joint.prey_reward.escape_streak = {}
                if hasattr(joint.prey_reward, 'position_history'):
                    joint.prey_reward.position_history = {}

            # 如果有v2_reward (标准版本)，也重置它
            if hasattr(joint, 'v2_reward') and hasattr(joint.v2_reward, 'reset'):
                joint.v2_reward.reset()
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
