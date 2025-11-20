"""状态数据模型"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict


EntityType = Literal["hunter", "prey"]


@dataclass
class RayHit:
    """射线碰撞信息"""
    angle: float
    distance: float
    hit_type: Optional[EntityType] = None
    hit_id: Optional[str] = None


@dataclass
class Event:
    """事件数据"""
    type: Literal["predation", "breed", "spawn", "despawn", "grow"]
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    energy_gain: Optional[float] = None
    parent_id: Optional[str] = None
    child: Optional[Dict] = None


@dataclass
class EntityState:
    """实体状态"""
    # 基本属性
    id: str
    type: EntityType
    x: float
    y: float
    angle: float
    speed: float
    angular_velocity: float
    radius: float = 10.0

    # 生命属性
    energy: float = 100.0
    digestion: float = 0.0
    age: float = 0.0
    generation: int = 0
    offspring_count: int = 0

    # 传感器属性
    fov_deg: Optional[float] = None
    fov_range: Optional[float] = None
    rays: List[RayHit] = field(default_factory=list)

    # 繁殖属性
    split_energy: float = 120.0
    breed_cd: float = 0.0
    spawn_progress: float = 1.0

    # 目标追踪
    target_id: Optional[str] = None
    iteration: int = 0

    # 其他属性
    should_persist: bool = False
    lifespan: Optional[float] = None
    saved: bool = False


@dataclass
class WorldState:
    """世界状态"""
    tick: int
    entities: List[EntityState] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    counters: Optional[Dict[str, int]] = None

    @staticmethod
    def from_dict(payload: Dict) -> WorldState:
        """从字典创建世界状态"""
        tick = int(payload.get("tick", 0))
        entities: List[EntityState] = []

        for e in payload.get("entities", []):
            rays = [RayHit(**r) for r in e.get("rays", [])]
            entities.append(
                EntityState(
                    id=str(e.get("id")),
                    type=str(e.get("type")),
                    x=float(e.get("x", 0.0)),
                    y=float(e.get("y", 0.0)),
                    angle=float(e.get("angle", 0.0)),
                    speed=float(e.get("speed", 0.0)),
                    angular_velocity=float(e.get("angular_velocity", 0.0)),
                    radius=float(e.get("radius", 10.0)),
                    energy=float(e.get("energy", 100.0)),
                    digestion=float(e.get("digestion", 0.0)),
                    age=float(e.get("age", 0.0)),
                    generation=int(e.get("generation", 0)),
                    offspring_count=int(e.get("offspring_count", 0)),
                    fov_deg=(
                        float(e["fov_deg"]) if e.get("fov_deg") is not None else None
                    ),
                    fov_range=(
                        float(e["fov_range"]) if e.get("fov_range") is not None else None
                    ),
                    rays=rays,
                    split_energy=float(e.get("split_energy", 120.0)),
                    target_id=e.get("target_id"),
                    iteration=int(e.get("iteration", tick)),
                    breed_cd=float(e.get("breed_cd", 0.0)),
                    spawn_progress=float(e.get("spawn_progress", 1.0)),
                    should_persist=bool(e.get("should_persist", False)),
                    lifespan=e.get("lifespan"),
                    saved=bool(e.get("saved", False)),
                )
            )

        events = [Event(**ev) for ev in payload.get("events", [])]
        counters = payload.get("counters")

        return WorldState(tick=tick, entities=entities, events=events, counters=counters)
