"""能量系统"""

from __future__ import annotations
import math
from typing import List
from models import EntityState, Event
from config import EnvConfig


class EnergySystem:
    """能量系统管理器"""

    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self.events: List[Event] = []

    def update_entity_energy(self, entity: EntityState, dt: float) -> None:
        """更新实体能量"""
        if entity.type == "prey":
            self._update_prey_energy(entity, dt)
        elif entity.type == "hunter":
            self._update_hunter_energy(entity, dt)

    def _update_prey_energy(self, entity: EntityState, dt: float) -> None:
        """更新猎物能量(生产者模式)"""
        # 能量生产
        energy_produced = (
            self.config.ENERGY_PRODUCTION_PREY
            * self.config.ENERGY_PRODUCTION_EFFICIENCY
            * dt
        )

        # 基础代谢
        metabolism_cost = self.config.ENERGY_BASE_METABOLISM_PREY * dt

        # 运动消耗
        normalized_speed = entity.speed / self.config.ENERGY_SPEED_REFERENCE
        movement_cost = self.config.ENERGY_MOVEMENT_COST * (normalized_speed**2) * dt

        # 转向消耗
        normalized_angular = abs(entity.angular_velocity) / math.pi
        turning_cost = self.config.ENERGY_TURNING_COST * normalized_angular * dt

        # 净能量变化
        net_energy_change = (
            energy_produced - metabolism_cost - movement_cost - turning_cost
        )

        # 更新能量
        old_energy = entity.energy
        entity.energy = min(
            entity.energy + net_energy_change, self.config.ENERGY_MAX_PREY
        )

        # 记录能量增长事件
        if entity.energy > old_energy and (entity.energy - old_energy) > 0.5:
            self.events.append(
                Event(
                    type="grow",
                    actor_id=entity.id,
                    energy_gain=entity.energy - old_energy,
                )
            )

        # 检查能量耗尽
        if entity.energy <= 0:
            self.events.append(Event(type="despawn", actor_id=entity.id))

    def _update_hunter_energy(self, entity: EntityState, dt: float) -> None:
        """更新猎人能量(消费者模式)"""
        # 基础代谢
        metabolism_cost = self.config.ENERGY_BASE_METABOLISM_HUNTER * dt

        # 运动消耗
        normalized_speed = entity.speed / self.config.ENERGY_SPEED_REFERENCE
        movement_cost = self.config.ENERGY_MOVEMENT_COST * (normalized_speed**2) * dt

        # 转向消耗
        normalized_angular = abs(entity.angular_velocity) / math.pi
        turning_cost = self.config.ENERGY_TURNING_COST * normalized_angular * dt

        # 总消耗
        total_cost = metabolism_cost + movement_cost + turning_cost

        # 更新能量
        entity.energy -= total_cost

        # 更新消化状态
        if entity.digestion > 0:
            entity.digestion = max(0, entity.digestion - dt)

        # 限制能量上限
        entity.energy = min(entity.energy, self.config.ENERGY_MAX_HUNTER)

        # 检查能量耗尽
        if entity.energy <= 0:
            self.events.append(Event(type="despawn", actor_id=entity.id))

    def process_predation(self, hunter: EntityState, prey: EntityState) -> bool:
        """处理捕食事件"""
        # 检查消化冷却
        if hunter.digestion > 0:
            return False

        # 计算距离
        dx = hunter.x - prey.x
        dy = hunter.y - prey.y
        distance = math.sqrt(dx * dx + dy * dy)

        # 计算捕食半径
        predation_radius = self._calculate_predation_radius(hunter)

        # 检查捕食范围
        if distance > predation_radius:
            return False

        # 执行捕食
        energy_gained = self.config.ENERGY_PREDATION_GAIN
        hunter.energy = min(
            hunter.energy + energy_gained, self.config.ENERGY_MAX_HUNTER
        )
        hunter.digestion = self.config.DIGESTION_DURATION

        # 记录事件
        self.events.append(
            Event(
                type="predation",
                actor_id=hunter.id,
                target_id=prey.id,
                energy_gain=energy_gained,
            )
        )

        self.events.append(Event(type="despawn", actor_id=prey.id))

        return True

    def _calculate_predation_radius(self, hunter: EntityState) -> float:
        """计算捕食半径"""
        from config import AgentConfig

        agent_cfg = AgentConfig()

        # 基础半径
        fov_range = hunter.fov_range or agent_cfg.HUNTER_FOV_RANGE
        base_radius = fov_range * self.config.PREDATION_BASE_RATIO

        # 体型加成
        size_factor = hunter.radius / self.config.DEFAULT_RADIUS
        size_bonus = 1 + (size_factor - 1) * self.config.PREDATION_SIZE_BONUS
        radius_with_size = base_radius * size_bonus

        # 速度加成
        speed_factor = hunter.speed / self.config.ENERGY_SPEED_REFERENCE
        speed_bonus = 1 + speed_factor * self.config.PREDATION_SPEED_BONUS
        final_radius = radius_with_size * speed_bonus

        # 限制范围
        return max(
            self.config.PREDATION_MIN_RADIUS,
            min(final_radius, self.config.PREDATION_MAX_RADIUS),
        )

    def check_breeding(self, entity: EntityState) -> bool:
        """检查是否可以繁殖"""
        if entity.energy < entity.split_energy:
            return False
        if entity.breed_cd > 0:
            return False
        return True

    def get_events(self) -> List[Event]:
        """获取事件列表"""
        return self.events.copy()

    def clear_events(self):
        """清空事件"""
        self.events.clear()

    def get_stats(self, entities: List[EntityState]) -> dict:
        """获取能量统计"""
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
