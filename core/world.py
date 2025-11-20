"""世界模拟器"""

from __future__ import annotations
import random
from typing import List, Optional
from models import EntityState, WorldState, Event
from config import EnvConfig, AgentConfig
from .energy import EnergySystem
from .physics import PhysicsEngine
from .sensors import SensorSystem


class WorldSimulator:
    """世界模拟器 - 整合所有后端系统"""

    def __init__(
        self,
        env_config: EnvConfig = None,
        agent_config: AgentConfig = None,
        use_parallel: bool = True,
    ):
        self.env_config = env_config or EnvConfig()
        self.agent_config = agent_config or AgentConfig()
        self.use_parallel = use_parallel

        # 初始化子系统
        self.energy_system = EnergySystem(self.env_config)
        self.physics_engine = PhysicsEngine(self.env_config)
        self.sensor_system = SensorSystem(use_parallel=use_parallel)

        # 世界状态
        self.tick = 0
        self.entities: List[EntityState] = []

    def initialize(self, n_hunters: int = 6, n_prey: int = 18):
        """初始化世界"""
        self.tick = 0
        self.entities.clear()

        # 生成初始实体
        for _ in range(n_hunters):
            entity = self.physics_engine.spawn_entity("hunter")
            self.entities.append(entity)

        for _ in range(n_prey):
            entity = self.physics_engine.spawn_entity("prey")
            self.entities.append(entity)

    def step(self) -> WorldState:
        """执行一个时间步"""
        self.tick += 1
        dt = self.env_config.DT

        # 1. 更新能量
        for entity in self.entities:
            self.energy_system.update_entity_energy(entity, dt)

        # 2. 更新物理
        self.physics_engine.update_motion(self.entities, dt)

        # 3. 移除死亡实体
        self._remove_dead_entities()

        # 4. 处理捕食
        self._handle_predation()

        # 5. 处理繁殖
        self._handle_breeding()

        # 6. 更新传感器
        self.sensor_system.update_all_rays(
            self.entities, self.agent_config.DEFAULT_RAY_COUNT
        )

        # 7. 更新迭代标记
        for entity in self.entities:
            entity.iteration = self.tick

        # 8. 获取事件
        events = self.energy_system.get_events()
        self.energy_system.clear_events()

        # 9. 深拷贝实体以避免状态共享 (修复奖励计算问题)
        import copy
        entities_snapshot = [copy.copy(e) for e in self.entities]

        return WorldState(tick=self.tick, entities=entities_snapshot, events=events)

    def _remove_dead_entities(self):
        """移除死亡的实体"""
        self.entities = [
            e for e in self.entities if not (e.type == "hunter" and e.energy <= 0)
        ]

    def _handle_predation(self):
        """处理捕食"""
        hunters = [e for e in self.entities if e.type == "hunter"]
        preys = [e for e in self.entities if e.type == "prey"]
        eaten_ids = set()

        if self.use_parallel and self.sensor_system.parallel_renderer:
            # 使用四叉树加速
            prey_map = {p.id: p for p in preys}

            for hunter in hunters:
                if hunter.digestion > 0:
                    continue

                # 查询附近的猎物
                nearby_ids = self.sensor_system.parallel_renderer.quadtree.query_circle(
                    hunter.x, hunter.y, hunter.fov_range * 0.2
                )

                nearby_preys = [
                    prey_map[pid] for pid in nearby_ids if pid in prey_map
                ]

                if nearby_preys:
                    target = random.choice(nearby_preys)
                    if self.energy_system.process_predation(hunter, target):
                        eaten_ids.add(target.id)
        else:
            # 暴力检测
            for hunter in hunters:
                if hunter.digestion > 0:
                    continue

                nearby_preys = [
                    p
                    for p in preys
                    if (p.x - hunter.x) ** 2 + (p.y - hunter.y) ** 2
                    < (hunter.fov_range * 0.2) ** 2
                ]

                if nearby_preys:
                    target = random.choice(nearby_preys)
                    if self.energy_system.process_predation(hunter, target):
                        eaten_ids.add(target.id)

        # 移除被吃掉的猎物
        if eaten_ids:
            self.entities = [e for e in self.entities if e.id not in eaten_ids]

    def _handle_breeding(self):
        """
        处理繁殖

        繁殖条件：
        1. 能量达到split_energy阈值
        2. 繁殖冷却时间breed_cd归零
        3. 未达到最大实体数量限制
        """
        if len(self.entities) >= self.env_config.MAX_ENTITIES:
            return

        new_entities = []

        # 猎人繁殖 - 仅检查能量和冷却时间
        for hunter in [e for e in self.entities if e.type == "hunter"]:
            if (
                self.energy_system.check_breeding(hunter)
                and len(self.entities) + len(new_entities) < self.env_config.MAX_ENTITIES
            ):
                child = self.physics_engine.spawn_entity("hunter", parent=hunter)
                hunter.energy *= 0.5
                hunter.offspring_count += 1
                hunter.breed_cd = self.env_config.BREED_CD_HUNTER
                new_entities.append(child)

        # 猎物繁殖 - 仅检查能量和冷却时间
        for prey in [e for e in self.entities if e.type == "prey"]:
            if (
                self.energy_system.check_breeding(prey)
                and len(self.entities) + len(new_entities) < self.env_config.MAX_ENTITIES
            ):
                child = self.physics_engine.spawn_entity("prey", parent=prey)
                prey.energy *= 0.5
                prey.offspring_count += 1
                prey.breed_cd = self.env_config.BREED_CD_PREY
                new_entities.append(child)

        self.entities.extend(new_entities)

    def get_stats(self) -> dict:
        """获取统计信息"""
        stats = self.energy_system.get_stats(self.entities)
        stats.update(self.sensor_system.get_stats())
        return stats

    def shutdown(self):
        """关闭模拟器"""
        self.sensor_system.shutdown()
