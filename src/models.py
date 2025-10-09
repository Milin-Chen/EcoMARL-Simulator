from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict


EntityType = Literal["hunter", "prey"]


@dataclass
class RayHit:
    angle: float
    distance: float
    hit_type: Optional[EntityType] = None
    hit_id: Optional[str] = None


@dataclass
class NeuralNetSpec:
    # 前端仅做展示占位；后端可传入结构与权重摘要
    layers: List[int] = field(default_factory=lambda: [8, 32, 2])
    has_bias: bool = True
    mutation: Optional[str] = None  # 如: weights_mod/new_conn/new_neuron


@dataclass
class Event:
    # 事件驱动：用于从后端JSON传递捕食/繁殖/成长等效果，由前端渲染接管逻辑
    type: Literal["predation", "breed", "spawn", "despawn", "grow"]
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    energy_gain: Optional[float] = None
    parent_id: Optional[str] = None
    child: Optional[Dict] = None  # 子体的基本字段（id/type/x/y/angle/radius等）


@dataclass
class EntityState:
    id: str
    type: EntityType
    x: float
    y: float
    angle: float  # 弧度
    speed: float  # 像素/秒
    angular_velocity: float  # 弧度/秒
    radius: float = 10.0

    energy: float = 100.0
    digestion: float = 0.0
    age: float = 0.0
    generation: int = 0
    offspring_count: int = 0

    # 视野参数交由前端配置主导；如后端提供则覆盖
    fov_deg: Optional[float] = None
    fov_range: Optional[float] = None
    rays: List[RayHit] = field(default_factory=list)

    nn: NeuralNetSpec = field(default_factory=NeuralNetSpec)

    # 前端渲染/调试用
    split_energy: float = 120.0
    target_id: Optional[str] = None
    iteration: int = 0
    # 繁殖冷却（秒），用于控制繁殖速率
    breed_cd: float = 0.0
    # 平滑分裂与持久化占位
    spawn_progress: float = 1.0  # 0..1，子体由小到大平滑长成
    should_persist: bool = False
    lifespan: Optional[float] = None  # 预留寿命（秒），None表示不限制
    saved: bool = False


@dataclass
class WorldState:
    tick: int
    entities: List[EntityState] = field(default_factory=list)
    # 新增：事件与计数（用于前端渲染驱动效果与展示）
    events: List[Event] = field(default_factory=list)
    counters: Optional[Dict[str, int]] = None

    @staticmethod
    def from_dict(payload: Dict) -> "WorldState":
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
                    # 若后端提供，则使用；否则由前端按类型配置
                    fov_deg=(float(e["fov_deg"]) if e.get("fov_deg") is not None else None),
                    fov_range=(float(e["fov_range"]) if e.get("fov_range") is not None else None),
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