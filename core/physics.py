"""物理引擎"""

from __future__ import annotations
import math
import random
import time
import threading
from typing import List
from models import EntityState
from config import EnvConfig


class PhysicsEngine:
    """物理引擎 - 处理运动和碰撞"""

    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self._entity_counter = 0
        self._counter_lock = threading.Lock()

    def update_motion(self, entities: List[EntityState], dt: float):
        """更新实体运动"""
        for entity in entities:
            # 猎物能量为0时不移动
            if entity.type == "prey" and entity.energy <= 0:
                continue

            # 更新角度和位置
            entity.angle += entity.angular_velocity * dt
            entity.x += math.cos(entity.angle) * entity.speed * dt
            entity.y += math.sin(entity.angle) * entity.speed * dt

            # 边界反弹
            self._handle_boundary_collision(entity)

            # 更新年龄和冷却
            entity.age += dt
            entity.breed_cd = max(0.0, entity.breed_cd - dt)
            entity.spawn_progress = min(1.0, entity.spawn_progress + self.config.SPAWN_GROW_RATE * dt)

    def _handle_boundary_collision(self, entity: EntityState):
        """处理边界碰撞"""
        if entity.x < entity.radius or entity.x > self.config.WINDOW_WIDTH - entity.radius:
            entity.angular_velocity *= -1
            entity.angle += math.pi / 2
            entity.x = max(entity.radius, min(entity.x, self.config.WINDOW_WIDTH - entity.radius))

        if entity.y < entity.radius or entity.y > self.config.WINDOW_HEIGHT - entity.radius:
            entity.angular_velocity *= -1
            entity.angle += math.pi / 2
            entity.y = max(entity.radius, min(entity.y, self.config.WINDOW_HEIGHT - entity.radius))

    def spawn_entity(
        self, entity_type: str, parent: EntityState = None
    ) -> EntityState:
        """生成新实体"""
        from config import AgentConfig

        agent_cfg = AgentConfig()

        if parent:
            # 从父实体繁殖
            entity_id = self._get_next_id(entity_type)
            angle = parent.angle + random.uniform(-0.25, 0.25)
            speed = max(6.0, parent.speed + random.uniform(-2.0, 2.0))
            radius = max(6.0, parent.radius + random.uniform(-0.6, 0.6))
            offset = parent.radius * 1.6

            x = self._clamp(
                parent.x + math.cos(angle) * offset,
                radius,
                self.config.WINDOW_WIDTH - radius,
            )
            y = self._clamp(
                parent.y + math.sin(angle) * offset,
                radius,
                self.config.WINDOW_HEIGHT - radius,
            )

            fov_deg = parent.fov_deg
            fov_range = parent.fov_range

            return EntityState(
                id=entity_id,
                type=entity_type,
                x=x,
                y=y,
                angle=angle,
                speed=speed,
                angular_velocity=max(
                    -0.8,
                    min(0.8, parent.angular_velocity + random.uniform(-0.2, 0.2)),
                ),
                radius=radius,
                energy=parent.energy * 0.5,
                digestion=0.0,
                age=0.0,
                generation=parent.generation + 1,
                offspring_count=0,
                fov_deg=fov_deg,
                fov_range=fov_range,
                split_energy=parent.split_energy,
                breed_cd=(
                    self.config.BREED_CD_PREY
                    if entity_type == "prey"
                    else self.config.BREED_CD_HUNTER
                ),
                spawn_progress=0.0,
            )
        else:
            # 随机生成
            entity_id = self._get_next_id(entity_type)
            x = random.uniform(40, self.config.WINDOW_WIDTH - 40)
            y = random.uniform(40, self.config.WINDOW_HEIGHT - 40)
            angle = random.uniform(-math.pi, math.pi)

            if entity_type == "hunter":
                speed = random.uniform(agent_cfg.HUNTER_SPEED_MIN, agent_cfg.HUNTER_SPEED_MAX)
                fov_deg = agent_cfg.HUNTER_FOV_DEG
                fov_range = agent_cfg.HUNTER_FOV_RANGE
            else:
                speed = random.uniform(agent_cfg.PREY_SPEED_MIN, agent_cfg.PREY_SPEED_MAX)
                fov_deg = agent_cfg.PREY_FOV_DEG
                fov_range = agent_cfg.PREY_FOV_RANGE

            # 根据实体类型设置繁殖能量阈值
            split_energy = (
                self.config.ENERGY_SPLIT_HUNTER
                if entity_type == "hunter"
                else self.config.ENERGY_SPLIT_PREY
            )

            return EntityState(
                id=entity_id,
                type=entity_type,
                x=x,
                y=y,
                angle=angle,
                speed=speed,
                angular_velocity=random.uniform(-0.8, 0.8),
                radius=self.config.DEFAULT_RADIUS,
                energy=random.uniform(60, 120),
                fov_deg=fov_deg,
                fov_range=fov_range,
                split_energy=split_energy,
            )

    def _get_next_id(self, entity_type: str) -> str:
        """生成唯一ID"""
        with self._counter_lock:
            self._entity_counter += 1
            entity_id = self._entity_counter

        timestamp_suffix = str(int(time.time() * 10000) % 10000).zfill(4)
        return f"{entity_type[0]}_{entity_id:06d}_{timestamp_suffix}"

    @staticmethod
    def _clamp(v: float, lo: float, hi: float) -> float:
        """限制值在范围内"""
        return max(lo, min(hi, v))
