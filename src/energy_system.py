"""
能量系统管理模块
处理实体的能量生产、消耗和相关逻辑
"""

from __future__ import annotations
import math
from typing import List, Optional
from models import EntityState, Event
import config


class EnergyManager:
    """能量系统管理器"""

    def __init__(self):
        self.events: List[Event] = []

    def update_entity_energy(self, entity: EntityState, dt: float) -> None:
        """
        更新单个实体的能量状态

        Args:
            entity: 实体对象
            dt: 时间增量（秒）
        """
        if entity.type == "prey":
            self._update_prey_energy(entity, dt)
        elif entity.type == "hunter":
            self._update_hunter_energy(entity, dt)

    def _update_prey_energy(self, entity: EntityState, dt: float) -> None:
        """
        更新猎物能量（生产者模式）

        猎物作为生产者，每秒产生能量，类似植物的光合作用
        """
        # 1. 能量生产（核心特性）
        energy_produced = (
            config.ENERGY_PRODUCTION_PREY * config.ENERGY_PRODUCTION_EFFICIENCY * dt
        )

        # 2. 基础代谢消耗
        metabolism_cost = config.ENERGY_BASE_METABOLISM_PREY * dt

        # 3. 运动能量消耗
        normalized_speed = entity.speed / config.ENERGY_SPEED_REFERENCE
        movement_cost = config.ENERGY_MOVEMENT_COST * (normalized_speed**2) * dt

        # 4. 转向能量消耗
        normalized_angular = abs(entity.angular_velocity) / math.pi
        turning_cost = config.ENERGY_TURNING_COST * normalized_angular * dt

        # 5. 计算净能量变化
        net_energy_change = (
            energy_produced - metabolism_cost - movement_cost - turning_cost
        )

        # 6. 更新能量（不超过上限）
        old_energy = entity.energy
        entity.energy = min(entity.energy + net_energy_change, config.ENERGY_MAX_PREY)

        # 7. 记录能量增长事件（用于可视化）
        if entity.energy > old_energy and (entity.energy - old_energy) > 0.5:
            self.events.append(
                Event(
                    type="grow",
                    actor_id=entity.id,
                    energy_gain=entity.energy - old_energy,
                )
            )

        # 8. 检查能量耗尽
        if entity.energy <= 0:
            self.events.append(Event(type="despawn", actor_id=entity.id))

    def _update_hunter_energy(self, entity: EntityState, dt: float) -> None:
        """
        更新猎人能量（消费者模式）

        猎人作为消费者，只能通过捕食获取能量
        """
        # 1. 基础代谢消耗
        metabolism_cost = config.ENERGY_BASE_METABOLISM_HUNTER * dt

        # 2. 运动能量消耗
        normalized_speed = entity.speed / config.ENERGY_SPEED_REFERENCE
        movement_cost = config.ENERGY_MOVEMENT_COST * (normalized_speed**2) * dt

        # 3. 转向能量消耗
        normalized_angular = abs(entity.angular_velocity) / math.pi
        turning_cost = config.ENERGY_TURNING_COST * normalized_angular * dt

        # 4. 总消耗
        total_cost = metabolism_cost + movement_cost + turning_cost

        # 5. 更新能量
        entity.energy -= total_cost

        # 6. 更新消化状态
        if entity.digestion > 0:
            entity.digestion = max(0, entity.digestion - dt)

        # 7. 限制能量上限
        entity.energy = min(entity.energy, config.ENERGY_MAX_HUNTER)

        # 8. 检查能量耗尽
        if entity.energy <= 0:
            self.events.append(Event(type="despawn", actor_id=entity.id))

    def process_predation(self, hunter: EntityState, prey: EntityState) -> bool:
        """
        处理捕食事件

        Args:
            hunter: 猎人实体
            prey: 猎物实体

        Returns:
            是否成功捕食
        """
        # 检查猎人是否在消化冷却中
        if hunter.digestion > 0:
            return False

        # 计算距离
        dx = hunter.x - prey.x
        dy = hunter.y - prey.y
        distance = math.sqrt(dx * dx + dy * dy)

        # 计算捕食半径
        predation_radius = self._calculate_predation_radius(hunter)

        # 检查是否在捕食范围内
        if distance > predation_radius:
            return False

        # 执行捕食
        energy_gained = config.ENERGY_PREDATION_GAIN
        hunter.energy = min(hunter.energy + energy_gained, config.ENERGY_MAX_HUNTER)
        hunter.digestion = config.DIGESTION_DURATION

        # 记录捕食事件
        self.events.append(
            Event(
                type="predation",
                actor_id=hunter.id,
                target_id=prey.id,
                energy_gain=energy_gained,
            )
        )

        # 猎物被捕食后死亡
        self.events.append(Event(type="despawn", actor_id=prey.id))

        return True

    def _calculate_predation_radius(self, hunter: EntityState) -> float:
        """
        计算猎人的捕食半径

        Args:
            hunter: 猎人实体

        Returns:
            捕食半径（像素）
        """
        # 基础半径
        fov_range = hunter.fov_range or config.HUNTER_FOV_RANGE
        base_radius = fov_range * config.PREDATION_BASE_RATIO

        # 体型加成
        size_factor = hunter.radius / config.DEFAULT_RADIUS
        size_bonus = 1 + (size_factor - 1) * config.PREDATION_SIZE_BONUS
        radius_with_size = base_radius * size_bonus

        # 速度加成
        speed_factor = hunter.speed / config.ENERGY_SPEED_REFERENCE
        speed_bonus = 1 + speed_factor * config.PREDATION_SPEED_BONUS
        final_radius = radius_with_size * speed_bonus

        # 限制在合理范围内
        return max(
            config.PREDATION_MIN_RADIUS, min(final_radius, config.PREDATION_MAX_RADIUS)
        )

    def check_breeding(self, entity: EntityState) -> bool:
        """
        检查实体是否可以繁殖

        Args:
            entity: 实体对象

        Returns:
            是否可以繁殖
        """
        # 检查能量是否达到繁殖阈值
        if entity.energy < entity.split_energy:
            return False

        # 检查繁殖冷却
        if entity.breed_cd > 0:
            return False

        return True

    def get_energy_stats(self, entities: List[EntityState]) -> dict:
        """
        获取能量系统统计信息

        Args:
            entities: 实体列表

        Returns:
            统计信息字典
        """
        hunters = [e for e in entities if e.type == "hunter"]
        preys = [e for e in entities if e.type == "prey"]

        stats = {
            "total_entities": len(entities),
            "hunters": len(hunters),
            "preys": len(preys),
        }

        if hunters:
            stats["hunter_avg_energy"] = sum(h.energy for h in hunters) / len(hunters)
            stats["hunter_max_energy"] = max(h.energy for h in hunters)
            stats["hunter_min_energy"] = min(h.energy for h in hunters)

        if preys:
            stats["prey_avg_energy"] = sum(p.energy for p in preys) / len(preys)
            stats["prey_max_energy"] = max(p.energy for p in preys)
            stats["prey_min_energy"] = min(p.energy for p in preys)

        return stats

    def clear_events(self):
        """清空事件列表"""
        self.events.clear()

    def get_events(self) -> List[Event]:
        """获取当前帧的所有事件"""
        return self.events.copy()


# 便捷函数
def update_all_entities_energy(
    entities: List[EntityState], dt: float, manager: Optional[EnergyManager] = None
) -> List[Event]:
    """
    更新所有实体的能量状态

    Args:
        entities: 实体列表
        dt: 时间增量（秒）
        manager: 能量管理器（可选，未提供则创建临时的）

    Returns:
        本次更新产生的事件列表
    """
    if manager is None:
        manager = EnergyManager()

    manager.clear_events()

    for entity in entities:
        manager.update_entity_energy(entity, dt)

    return manager.get_events()
