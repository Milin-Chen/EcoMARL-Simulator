"""观察空间定义"""

from __future__ import annotations
import numpy as np
import math
from typing import List
from models import EntityState
from config import AgentConfig


class ObservationSpace:
    """观察空间"""

    def __init__(self, agent_config: AgentConfig = None):
        self.cfg = agent_config or AgentConfig()

    def get_observation(
        self, entity: EntityState, all_entities: List[EntityState]
    ) -> np.ndarray:
        """
        获取实体的观察

        观察包括:
        1. 自身状态: [能量归一化, 速度归一化, 角速度归一化, 消化状态]
        2. 射线信息: 每条射线 [距离归一化, 是否命中同类, 是否命中异类]
        3. 相对位置: [最近同类相对x, 最近同类相对y, 最近异类相对x, 最近异类相对y]
        """
        obs_parts = []

        # 1. 自身状态 (4维)
        energy_norm = entity.energy / 100.0  # 归一化到[0, 1]
        speed_norm = entity.speed / 60.0  # 假设最大速度60
        angular_vel_norm = entity.angular_velocity / (math.pi)  # 归一化到[-1, 1]
        digestion_norm = entity.digestion / 5.0  # 假设最大消化时间5秒

        obs_parts.extend([energy_norm, speed_norm, angular_vel_norm, digestion_norm])

        # 2. 射线信息 (ray_count * 3 维)
        ray_count = self.cfg.DEFAULT_RAY_COUNT
        fov_range = (
            self.cfg.HUNTER_FOV_RANGE
            if entity.type == "hunter"
            else self.cfg.PREY_FOV_RANGE
        )

        if entity.rays and len(entity.rays) == ray_count:
            for ray in entity.rays:
                distance_norm = ray.distance / fov_range
                hit_same = 1.0 if ray.hit_type == entity.type else 0.0
                hit_other = (
                    1.0
                    if ray.hit_type is not None and ray.hit_type != entity.type
                    else 0.0
                )
                obs_parts.extend([distance_norm, hit_same, hit_other])
        else:
            # 填充默认值
            obs_parts.extend([1.0, 0.0, 0.0] * ray_count)

        # 3. 相对位置信息 (4维)
        same_type = [e for e in all_entities if e.type == entity.type and e.id != entity.id]
        other_type = [e for e in all_entities if e.type != entity.type]

        # 最近同类
        if same_type:
            nearest_same = min(
                same_type,
                key=lambda e: (e.x - entity.x) ** 2 + (e.y - entity.y) ** 2,
            )
            rel_same_x = (nearest_same.x - entity.x) / fov_range
            rel_same_y = (nearest_same.y - entity.y) / fov_range
        else:
            rel_same_x, rel_same_y = 0.0, 0.0

        # 最近异类
        if other_type:
            nearest_other = min(
                other_type,
                key=lambda e: (e.x - entity.x) ** 2 + (e.y - entity.y) ** 2,
            )
            rel_other_x = (nearest_other.x - entity.x) / fov_range
            rel_other_y = (nearest_other.y - entity.y) / fov_range
        else:
            rel_other_x, rel_other_y = 0.0, 0.0

        obs_parts.extend([rel_same_x, rel_same_y, rel_other_x, rel_other_y])

        return np.array(obs_parts, dtype=np.float32)

    def get_space_info(self) -> dict:
        """获取观察空间信息"""
        ray_count = self.cfg.DEFAULT_RAY_COUNT
        obs_dim = 4 + ray_count * 3 + 4  # 自身状态 + 射线 + 相对位置

        return {
            "observation_dim": obs_dim,
            "self_state_dim": 4,
            "ray_dim": ray_count * 3,
            "relative_position_dim": 4,
        }
