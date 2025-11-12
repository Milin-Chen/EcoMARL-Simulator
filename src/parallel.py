"""
多线程渲染优化模块
使用线程池并行处理射线检测、碰撞检测等计算密集任务
"""

from __future__ import annotations
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Dict, Set
from dataclasses import dataclass
import threading

from models import EntityState, RayHit
from quadtree import QuadTree
import config


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
        """
        初始化并行渲染器

        Args:
            num_workers: 工作线程数,None则自动设为CPU核心数
        """
        import multiprocessing

        self.num_workers = num_workers or max(2, multiprocessing.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.quadtree = QuadTree(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
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
        """更新四叉树(主线程调用)"""
        start = time.perf_counter()

        self.quadtree.clear()
        self.entity_map.clear()

        for entity in entities:
            self.quadtree.insert(entity.id, entity.x, entity.y, entity.radius)
            self.entity_map[entity.id] = entity

        self.stats["quadtree_build_time"] = time.perf_counter() - start
        self.stats["entities_processed"] = len(entities)

    def raycast_entity(self, task: RaycastTask) -> RaycastResult:
        """
        为单个实体执行射线检测(工作线程调用)

        Args:
            task: 射线检测任务

        Returns:
            射线检测结果
        """
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

                # 获取实体信息(线程安全)
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
        self, entities: List[EntityState], ray_count: int = config.DEFAULT_RAY_COUNT
    ) -> Dict[str, List[RayHit]]:
        """
        并行计算所有实体的射线检测

        Args:
            entities: 实体列表
            ray_count: 每个实体的射线数量

        Returns:
            字典: entity_id -> rays
        """
        start = time.perf_counter()

        # 构建任务列表
        tasks = []
        for entity in entities:
            # 确定FOV参数
            if config.USE_ENTITY_FOV and entity.fov_deg is not None:
                fov_deg = entity.fov_deg
                fov_range = entity.fov_range or config.DEFAULT_FOV_RANGE_HUNTER
            else:
                if entity.type == "hunter":
                    fov_deg = config.HUNTER_FOV_DEG
                    fov_range = config.HUNTER_FOV_RANGE
                else:
                    fov_deg = config.PREY_FOV_DEG
                    fov_range = config.PREY_FOV_RANGE

            tasks.append(
                RaycastTask(
                    entity_id=entity.id,
                    x=entity.x,
                    y=entity.y,
                    angle=entity.angle,
                    fov_deg=fov_deg,
                    fov_range=fov_range,
                    ray_count=ray_count,
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

    def find_collisions_parallel(
        self, entities: List[EntityState], batch_size: int = 10
    ) -> List[Tuple[str, str]]:
        """
        并行查找所有碰撞对

        Args:
            entities: 实体列表
            batch_size: 每个批次处理的实体数

        Returns:
            碰撞对列表: [(id1, id2), ...]
        """
        start = time.perf_counter()

        def check_batch(batch: List[EntityState]) -> List[Tuple[str, str]]:
            """检测一批实体的碰撞"""
            collisions = []
            for entity in batch:
                # 查询附近的实体
                nearby_ids = self.quadtree.query_circle(
                    entity.x, entity.y, entity.radius * 3
                )

                for other_id in nearby_ids:
                    if other_id == entity.id or other_id < entity.id:
                        continue

                    other = self.entity_map.get(other_id)
                    if not other:
                        continue

                    # 精确碰撞检测
                    dx = entity.x - other.x
                    dy = entity.y - other.y
                    dist_sq = dx * dx + dy * dy
                    min_dist = entity.radius + other.radius

                    if dist_sq < min_dist * min_dist:
                        collisions.append((entity.id, other.id))

            return collisions

        # 分批处理
        batches = [
            entities[i : i + batch_size] for i in range(0, len(entities), batch_size)
        ]
        futures = [self.executor.submit(check_batch, batch) for batch in batches]

        all_collisions = []
        for future in as_completed(futures):
            try:
                all_collisions.extend(future.result())
            except Exception as e:
                print(f"碰撞检测异常: {e}")

        self.stats["collision_time"] = time.perf_counter() - start
        return all_collisions

    def query_visible_entities(
        self,
        camera_x: float,
        camera_y: float,
        camera_width: float,
        camera_height: float,
        margin: float = 100.0,
    ) -> Set[str]:
        """
        查询相机视野内的实体(带边距)

        Args:
            camera_x, camera_y: 相机左上角坐标
            camera_width, camera_height: 相机视野大小
            margin: 边距(确保运动的实体不会突然消失)

        Returns:
            可见实体ID集合
        """
        return self.quadtree.query_range(
            camera_x - margin,
            camera_y - margin,
            camera_width + 2 * margin,
            camera_height + 2 * margin,
        )

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


class AdaptiveTaskScheduler:
    """自适应任务调度器 - 根据负载动态调整并行策略"""

    def __init__(self, renderer: ParallelRenderer):
        self.renderer = renderer
        self.history_times = []
        self.max_history = 30
        self.parallel_threshold = 20  # 实体数阈值

    def should_parallelize(self, num_entities: int) -> bool:
        """判断是否应该使用并行计算"""
        # 实体数较少时,串行更快(避免线程开销)
        if num_entities < self.parallel_threshold:
            return False

        # 基于历史性能动态调整
        if len(self.history_times) >= 10:
            avg_time = sum(self.history_times[-10:]) / 10
            # 如果平均处理时间很短,可能串行更快
            if avg_time < 0.001:  # 1ms
                return False

        return True

    def record_time(self, elapsed: float):
        """记录执行时间"""
        self.history_times.append(elapsed)
        if len(self.history_times) > self.max_history:
            self.history_times.pop(0)

    def get_optimal_batch_size(self, num_entities: int) -> int:
        """获取最优批次大小"""
        # 根据实体数量和工作线程数计算
        workers = self.renderer.num_workers
        return max(5, min(20, num_entities // (workers * 2)))
