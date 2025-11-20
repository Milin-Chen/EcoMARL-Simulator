"""并行渲染器"""

from __future__ import annotations
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict
from dataclasses import dataclass
import threading

from models import EntityState, RayHit
from .quadtree import QuadTree


@dataclass
class RaycastTask:
    """射线检测任务"""
    entity_id: str
    x: float
    y: float
    angle: float
    fov_deg: float
    fov_range: float
    ray_count: int


@dataclass
class RaycastResult:
    """射线检测结果"""
    entity_id: str
    rays: List[RayHit]


class ParallelRenderer:
    """并行渲染管理器"""

    def __init__(self, num_workers: Optional[int] = None):
        import multiprocessing
        from config import EnvConfig

        env_cfg = EnvConfig()

        self.num_workers = num_workers or max(2, multiprocessing.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.quadtree = QuadTree(env_cfg.WINDOW_WIDTH, env_cfg.WINDOW_HEIGHT)
        self.entity_map: Dict[str, EntityState] = {}
        self.lock = threading.Lock()

        # 性能统计
        self.stats = {
            "quadtree_build_time": 0.0,
            "raycast_time": 0.0,
            "collision_time": 0.0,
            "total_rays": 0,
            "entities_processed": 0,
        }

    def update_quadtree(self, entities: List[EntityState]):
        """更新四叉树"""
        start = time.perf_counter()

        self.quadtree.clear()
        self.entity_map.clear()

        for entity in entities:
            self.quadtree.insert(entity.id, entity.x, entity.y, entity.radius)
            self.entity_map[entity.id] = entity

        self.stats["quadtree_build_time"] = time.perf_counter() - start
        self.stats["entities_processed"] = len(entities)

    def raycast_entity(self, task: RaycastTask) -> RaycastResult:
        """为单个实体执行射线检测"""
        rays: List[RayHit] = []
        half = math.radians(task.fov_deg) / 2.0
        start_angle = task.angle - half
        step = (half * 2) / max(1, task.ray_count - 1)

        # 查询视野范围内的潜在碰撞对象
        fov_radius = task.fov_range
        nearby_ids = self.quadtree.query_circle(task.x, task.y, fov_radius)

        # 为每条射线计算碰撞
        for i in range(task.ray_count):
            ray_angle = start_angle + i * step
            min_dist = task.fov_range
            hit_type = None
            hit_id = None

            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)

            # 只检测视野范围内的实体
            for obj_id in nearby_ids:
                if obj_id == task.entity_id:
                    continue

                entity = self.entity_map.get(obj_id)
                if not entity:
                    continue

                # 射线与圆的相交检测
                ox = entity.x - task.x
                oy = entity.y - task.y

                # 投影到射线方向
                proj = ox * dx + oy * dy
                if proj < 0:
                    continue

                # 计算最近点到圆心的距离
                closest_x = ox - proj * dx
                closest_y = oy - proj * dy
                d2 = closest_x * closest_x + closest_y * closest_y

                if d2 <= entity.radius**2:
                    # 计算交点距离
                    dist = proj - math.sqrt(entity.radius**2 - d2)
                    if 0 < dist < min_dist:
                        min_dist = dist
                        hit_type = entity.type
                        hit_id = entity.id

            rays.append(
                RayHit(
                    angle=ray_angle, distance=min_dist, hit_type=hit_type, hit_id=hit_id
                )
            )

        return RaycastResult(entity_id=task.entity_id, rays=rays)

    def compute_rays_parallel(
        self, entities: List[EntityState], ray_count: int = 24, selected_id: str = None
    ) -> Dict[str, List[RayHit]]:
        """
        并行计算所有实体的射线检测

        Args:
            entities: 实体列表
            ray_count: 基础射线数量
            selected_id: 选中实体ID (用于LOD优化)
        """
        from config import AgentConfig

        agent_cfg = AgentConfig()
        start = time.perf_counter()

        # 构建任务列表
        tasks = []
        for entity in entities:
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

            # 性能优化: LOD射线数量 - 选中实体使用全部射线，其他实体使用一半
            entity_ray_count = ray_count
            if selected_id is not None and entity.id != selected_id:
                # 非选中实体: 减少射线数量 (18 → 9, 节省50%计算)
                entity_ray_count = max(8, ray_count // 2)

            tasks.append(
                RaycastTask(
                    entity_id=entity.id,
                    x=entity.x,
                    y=entity.y,
                    angle=entity.angle,
                    fov_deg=fov_deg,
                    fov_range=fov_range,
                    ray_count=entity_ray_count,
                )
            )

        # 并行执行射线检测
        results = {}
        futures = [self.executor.submit(self.raycast_entity, task) for task in tasks]

        for future in as_completed(futures):
            try:
                result = future.result()
                results[result.entity_id] = result.rays
            except Exception as e:
                print(f"射线检测异常: {e}")

        elapsed = time.perf_counter() - start
        self.stats["raycast_time"] = elapsed
        self.stats["total_rays"] = len(tasks) * ray_count

        return results

    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        qt_stats = self.quadtree.get_stats()
        return {
            **self.stats,
            **qt_stats,
            "num_workers": self.num_workers,
            "rays_per_second": (
                self.stats["total_rays"] / self.stats["raycast_time"]
                if self.stats["raycast_time"] > 0
                else 0
            ),
        }

    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)
