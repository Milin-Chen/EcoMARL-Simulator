from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict
from typing import Optional, List

from models import WorldState, EntityState, RayHit
import config
from parallel import ParallelRenderer, AdaptiveTaskScheduler


class DataSource:
    def poll(self) -> Optional[WorldState]:
        raise NotImplementedError


class MockSource(DataSource):
    """用于演示与前端调试的随机数据源(多线程+四叉树优化版)"""

    def __init__(self, n_hunters: int = 6, n_prey: int = 18, use_parallel: bool = True):
        self.tick = 0
        self.entities: List[EntityState] = []

        # 初始化并行渲染器
        self.use_parallel = use_parallel
        if use_parallel:
            self.renderer = ParallelRenderer()
            self.scheduler = AdaptiveTaskScheduler(self.renderer)
            print(f"✓ 并行渲染器已启用 (工作线程: {self.renderer.num_workers})")
        else:
            self.renderer = None
            self.scheduler = None
            print("⚠ 使用串行模式")

        def spawn_entity(idx: int, etype: str) -> EntityState:
            x = random.uniform(40, config.WINDOW_WIDTH - 40)
            y = random.uniform(40, config.WINDOW_HEIGHT - 40)
            angle = random.uniform(-math.pi, math.pi)
            speed = (
                random.uniform(20.0, 60.0)
                if etype == "hunter"
                else random.uniform(15.0, 40.0)
            )
            av = random.uniform(-0.8, 0.8)
            r = (
                config.DEFAULT_RADIUS
                if etype == "hunter"
                else config.DEFAULT_RADIUS * 0.9
            )
            fov_deg = (
                config.DEFAULT_FOV_DEG_HUNTER
                if etype == "hunter"
                else config.DEFAULT_FOV_DEG_PREY
            )
            fov_range = (
                config.DEFAULT_FOV_RANGE_HUNTER
                if etype == "hunter"
                else config.DEFAULT_FOV_RANGE_PREY
            )
            return EntityState(
                id=f"{etype[0]}_{idx}",
                type=etype,
                x=x,
                y=y,
                angle=angle,
                speed=speed,
                angular_velocity=av,
                radius=r,
                energy=random.uniform(60, 120),
                digestion=0.0,
                age=0.0,
                generation=0,
                offspring_count=0,
                fov_deg=fov_deg,
                fov_range=fov_range,
            )

        for i in range(n_hunters):
            self.entities.append(spawn_entity(i, "hunter"))
        for i in range(n_prey):
            self.entities.append(spawn_entity(i, "prey"))

    def _spawn_child(self, parent: EntityState, etype: Optional[str] = None):
        # 简化分裂：幅度更小，尽量保持父体的行为模式
        etype = etype or parent.type
        nid = f"{etype[0]}_{int(time.time()*1000)}_{random.randint(10,99)}"
        angle = parent.angle + random.uniform(-0.25, 0.25)
        speed = max(6.0, parent.speed + random.uniform(-2.0, 2.0))
        r = max(6.0, parent.radius + random.uniform(-0.6, 0.6))
        off = parent.radius * 1.6
        x = clamp(
            parent.x + math.cos(angle) * off,
            parent.radius,
            config.WINDOW_WIDTH - parent.radius,
        )
        y = clamp(
            parent.y + math.sin(angle) * off,
            parent.radius,
            config.WINDOW_HEIGHT - parent.radius,
        )
        child = EntityState(
            id=nid,
            type=etype,
            x=x,
            y=y,
            angle=angle,
            speed=speed,
            angular_velocity=max(
                -0.8, min(0.8, parent.angular_velocity + random.uniform(-0.2, 0.2))
            ),
            radius=r,
            energy=parent.energy * 0.5,
            digestion=0.0,
            age=0.0,
            generation=parent.generation + 1,
            offspring_count=0,
            fov_deg=parent.fov_deg,
            fov_range=parent.fov_range,
            split_energy=parent.split_energy,
            breed_cd=(
                config.BREED_CD_PREY if etype == "prey" else config.BREED_CD_HUNTER
            ),
            spawn_progress=0.0,
            should_persist=False,
        )
        parent.energy *= 0.5
        parent.offspring_count += 1
        # 父体进入繁殖冷却，保证持续分裂但受冷却控制
        parent.breed_cd = (
            config.BREED_CD_PREY if parent.type == "prey" else config.BREED_CD_HUNTER
        )
        if len(self.entities) < config.MAX_ENTITIES:
            self.entities.append(child)

        return child

    def force_breed(self, entity_id: str) -> Optional[EntityState]:
        """外部接口：强制让指定实体立即分裂一个子体，并标记代数/计数。
        - 遵守最大实体数量限制
        - 父体与子体按类型应用繁殖冷却
        - 返回新生成的子体；若未找到实体或数量达上限则返回 None
        """
        if len(self.entities) >= config.MAX_ENTITIES:
            return None
        parent = next((e for e in self.entities if e.id == entity_id), None)
        if not parent:
            return None
        return self._spawn_child(parent, etype=parent.type)

    def _update_motion(self, dt: float):
        # 简单运动学更新（前端演示用）
        survivors = []
        for e in self.entities:
            # 能量衰减与年龄增长
            decay = 0.8 if e.type == "hunter" else 0.15
            e.energy = max(0.0, e.energy - decay * dt)
            e.age += dt
            # 繁殖冷却递减
            e.breed_cd = max(0.0, float(getattr(e, "breed_cd", 0.0)) - dt)
            # 平滑分裂成长（子体半径比例从SPAWN_MIN_SCALE向1.0增长）
            e.spawn_progress = min(
                1.0,
                float(getattr(e, "spawn_progress", 1.0)) + config.SPAWN_GROW_RATE * dt,
            )

            # 运动：猎物在能量为0时原地不动；捕食者能量为0时死亡
            if e.type == "hunter" and e.energy <= 0:
                # 捕食者死亡：不加入渲染与后续
                continue
            else:
                # 正常/猎物零能量时保持位置
                if not (e.type == "prey" and e.energy <= 0.0):
                    e.angle += e.angular_velocity * dt
                    e.x += math.cos(e.angle) * e.speed * dt
                    e.y += math.sin(e.angle) * e.speed * dt

            # 边界反弹（只对运动中的实体）
            if not (e.type == "prey" and e.energy <= 0.0):
                if e.x < e.radius or e.x > config.WINDOW_WIDTH - e.radius:
                    e.angular_velocity *= -1
                    e.angle += math.pi / 2
                if e.y < e.radius or e.y > config.WINDOW_HEIGHT - e.radius:
                    e.angular_velocity *= -1
                    e.angle += math.pi / 2

            survivors.append(e)

        self.entities = survivors

        # 偶发"猎人捕食猎物"事件（使用四叉树优化）
        hunters = [x for x in self.entities if x.type == "hunter"]
        preys = [x for x in self.entities if x.type == "prey"]
        eaten_ids = set()

        # 如果启用并行渲染，使用四叉树加速邻近查询
        if self.use_parallel and self.renderer:
            prey_map = {p.id: p for p in preys}

            for h in hunters:
                if h.digestion > 0:
                    h.digestion = max(0.0, h.digestion - dt)
                    continue

                # 使用四叉树查询附近的猎物
                nearby_ids = self.renderer.quadtree.query_circle(
                    h.x, h.y, h.fov_range * 0.2
                )

                near = [prey_map[pid] for pid in nearby_ids if pid in prey_map]

                if near:
                    target = random.choice(near)
                    h.energy = min(160.0, h.energy + 50.0)
                    h.digestion = 3.5
                    eaten_ids.add(target.id)

                # 猎人繁殖
                if (
                    h.breed_cd <= 0.0
                    and h.energy > h.split_energy
                    and random.random() < 0.05
                ):
                    self._spawn_child(h, etype="hunter")
        else:
            # 原始方法（无四叉树）
            for h in hunters:
                near = [
                    p
                    for p in preys
                    if (p.x - h.x) ** 2 + (p.y - h.y) ** 2 < (h.fov_range * 0.2) ** 2
                ]
                if near and h.digestion <= 0:
                    target = random.choice(near)
                    h.energy = min(160.0, h.energy + 50.0)
                    h.digestion = 3.5
                    eaten_ids.add(target.id)
                else:
                    h.digestion = max(0.0, h.digestion - dt)
                if (
                    h.breed_cd <= 0.0
                    and h.energy > h.split_energy
                    and random.random() < 0.05
                ):
                    self._spawn_child(h, etype="hunter")

        if eaten_ids:
            self.entities = [e for e in self.entities if e.id not in eaten_ids]

        # 猎物分裂：生存时间够久或能量达到分裂阈值
        for p in preys:
            if p.breed_cd <= 0.0 and (
                p.age > 8.0 or (p.energy > p.split_energy and random.random() < 0.12)
            ):
                if len(self.entities) < config.MAX_ENTITIES:
                    self._spawn_child(p, etype="prey")

    def _compute_rays(self, e: EntityState) -> List[RayHit]:
        # 如果后端不提供，前端近似计算射线与最近碰撞
        rays: List[RayHit] = []
        count = config.DEFAULT_RAY_COUNT
        half = math.radians(e.fov_deg) / 2.0
        start = e.angle - half
        step = (half * 2) / max(1, count - 1)
        for i in range(count):
            a = start + i * step
            min_dist = e.fov_range
            hit_type = None
            hit_id = None
            # 与其他实体圆形近似碰撞
            for o in self.entities:
                if o.id == e.id:
                    continue
                # 射线参数化：e -> e + t * dir
                dx = math.cos(a)
                dy = math.sin(a)
                ox = o.x - e.x
                oy = o.y - e.y
                # 最近点到圆中心的距离（几何近似）
                proj = ox * dx + oy * dy
                if proj < 0:
                    continue
                closest_x = ox - proj * dx
                closest_y = oy - proj * dy
                d2 = closest_x**2 + closest_y**2
                if d2 <= o.radius**2:
                    # 命中，计算沿射线的距离
                    dist = proj
                    if 0 < dist < min_dist:
                        min_dist = dist
                        hit_type = o.type
                        hit_id = o.id
            rays.append(
                RayHit(angle=a, distance=min_dist, hit_type=hit_type, hit_id=hit_id)
            )
        return rays

    def poll(self) -> Optional[WorldState]:
        self.tick += 1
        dt = 1 / 60.0
        self._update_motion(dt)

        # 使用并行或串行射线检测
        if self.use_parallel and self.renderer:
            # 更新四叉树
            self.renderer.update_quadtree(self.entities)

            # 根据实体数量决定是否并行
            if self.scheduler.should_parallelize(len(self.entities)):
                # 并行计算射线
                ray_results = self.renderer.compute_rays_parallel(
                    self.entities, config.DEFAULT_RAY_COUNT
                )

                # 应用结果
                for e in self.entities:
                    if e.id in ray_results:
                        e.rays = ray_results[e.id]
                    e.iteration = self.tick
            else:
                # 实体数量少时使用串行（避免线程开销）
                for e in self.entities:
                    e.rays = self._compute_rays(e)
                    e.iteration = self.tick
        else:
            # 原始串行方法
            for e in self.entities:
                e.rays = self._compute_rays(e)
                e.iteration = self.tick

        return WorldState(tick=self.tick, entities=self.entities)

    def get_performance_stats(self) -> dict:
        """获取性能统计（用于调试）"""
        if self.renderer:
            return self.renderer.get_performance_stats()
        return {}

    def shutdown(self):
        """清理资源"""
        if self.renderer:
            self.renderer.shutdown()


class FileJSONSource(DataSource):
    """从 JSON 文件读取世界状态，便于与后端(pymunk)联调。"""

    def __init__(self, path: str = "runtime/world.json"):
        self.path = path
        self._last_mtime = 0.0
        # 允许后端尚未创建文件
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def poll(self) -> Optional[WorldState]:
        try:
            mtime = os.path.getmtime(self.path)
            if mtime <= self._last_mtime:
                return None
            self._last_mtime = mtime
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return WorldState.from_dict(payload)
        except FileNotFoundError:
            return None
        except Exception:
            # 读取错误时忽略，保持上一帧
            return None


# WebSocketSource 占位：如果需要，可安装 websockets 并实现
class WebSocketSource(DataSource):
    def __init__(self, url: str = "ws://localhost:8765"):
        self.url = url
        self.connected = False

    def poll(self) -> Optional[WorldState]:
        # 为避免额外依赖，这里仅提供接口形状
        # 建议后续采用 asyncio + websockets 实现
        return None


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
