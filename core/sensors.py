"""传感器系统"""

from __future__ import annotations
import math
from typing import List, Optional, TYPE_CHECKING
from models import EntityState, RayHit

if TYPE_CHECKING:
    from parallel import ParallelRenderer


class SensorSystem:
    """传感器系统 - 处理射线检测"""

    def __init__(self, use_parallel: bool = False):
        self.use_parallel = use_parallel
        self.parallel_renderer: Optional[ParallelRenderer] = None

        if use_parallel:
            from parallel import ParallelRenderer
            self.parallel_renderer = ParallelRenderer()

    def compute_rays(
        self, entity: EntityState, entities: List[EntityState], ray_count: int = 24
    ) -> List[RayHit]:
        """计算单个实体的射线检测"""
        from config import AgentConfig

        agent_cfg = AgentConfig()

        # 确定FOV参数
        if agent_cfg.USE_ENTITY_FOV and entity.fov_deg is not None:
            fov_deg = entity.fov_deg
            fov_range = entity.fov_range or agent_cfg.HUNTER_FOV_RANGE
        else:
            if entity.type == "hunter":
                fov_deg = agent_cfg.HUNTER_FOV_DEG
                fov_range = agent_cfg.HUNTER_FOV_RANGE
            else:
                fov_deg = agent_cfg.PREY_FOV_DEG
                fov_range = agent_cfg.PREY_FOV_RANGE

        rays: List[RayHit] = []
        half = math.radians(fov_deg) / 2.0
        start = entity.angle - half
        step = (half * 2) / max(1, ray_count - 1)

        for i in range(ray_count):
            angle = start + i * step
            min_dist = fov_range
            hit_type = None
            hit_id = None

            # 射线方向
            dx = math.cos(angle)
            dy = math.sin(angle)

            # 检测与其他实体的碰撞
            for other in entities:
                if other.id == entity.id:
                    continue

                # 射线与圆的相交检测
                ox = other.x - entity.x
                oy = other.y - entity.y

                # 投影到射线方向
                proj = ox * dx + oy * dy
                if proj < 0:
                    continue

                # 计算最近点到圆心的距离
                closest_x = ox - proj * dx
                closest_y = oy - proj * dy
                d2 = closest_x * closest_x + closest_y * closest_y

                if d2 <= other.radius**2:
                    # 计算交点距离
                    dist = proj - math.sqrt(other.radius**2 - d2)
                    if 0 < dist < min_dist:
                        min_dist = dist
                        hit_type = other.type
                        hit_id = other.id

            rays.append(
                RayHit(angle=angle, distance=min_dist, hit_type=hit_type, hit_id=hit_id)
            )

        return rays

    def update_all_rays(self, entities: List[EntityState], ray_count: int = 24, selected_id: str = None):
        """
        更新所有实体的射线检测

        Args:
            entities: 实体列表
            ray_count: 射线数量
            selected_id: 选中实体ID (用于LOD优化，可选)
        """
        if self.use_parallel and self.parallel_renderer:
            # 使用并行计算
            self.parallel_renderer.update_quadtree(entities)
            ray_results = self.parallel_renderer.compute_rays_parallel(
                entities, ray_count, selected_id=selected_id  # 传递selected_id用于LOD
            )

            for entity in entities:
                if entity.id in ray_results:
                    entity.rays = ray_results[entity.id]
        else:
            # 串行计算
            for entity in entities:
                entity.rays = self.compute_rays(entity, entities, ray_count)

    def shutdown(self):
        """关闭传感器系统"""
        if self.parallel_renderer:
            self.parallel_renderer.shutdown()

    def get_stats(self) -> dict:
        """获取性能统计"""
        if self.parallel_renderer:
            return self.parallel_renderer.get_performance_stats()
        return {}
