# 核心模块详解

**Core Modules - 物理引擎与数据模型**

---

## 📋 目录

- [1. 概述](#1-概述)
- [2. WorldSimulator - 世界模拟器](#2-worldsimulator---世界模拟器)
- [3. PhysicsEngine - 物理引擎](#3-physicsengine---物理引擎)
- [4. SensorSystem - 传感器系统](#4-sensorsystem---传感器系统)
- [5. EnergySystem - 能量系统](#5-energysystem---能量系统)
- [6. 数据模型](#6-数据模型)
- [7. 使用示例](#7-使用示例)

---

## 1. 概述

核心模块（`core/`）是整个项目的物理引擎，负责模拟生态系统的所有底层逻辑。

### 1.1 架构图

```
┌─────────────────────────────────────┐
│      WorldSimulator (world.py)      │
│  统筹所有子系统，管理模拟循环       │
└──────┬──────────────────────────────┘
       │
       ├─► PhysicsEngine (physics.py)
       │   运动更新、碰撞检测
       │
       ├─► SensorSystem (sensors.py)
       │   视野计算、目标检测
       │
       ├─► EnergySystem (energy.py)
       │   能量消耗、死亡判定
       │
       └─► ParallelRenderer (parallel/)
           空间索引、并行加速
```

### 1.2 核心特性

| 特性 | 实现 | 性能 |
|------|------|------|
| **运动物理** | 速度、角速度、摩擦力 | ~60 FPS (100实体) |
| **碰撞检测** | QuadTree空间索引 | O(log n) |
| **视野系统** | 扇形视野 + 视距限制 | 并行计算 |
| **能量管理** | 消耗/补充/死亡 | 每帧 O(n) |

---

## 2. WorldSimulator - 世界模拟器

**文件**: `core/world.py`

### 2.1 功能

WorldSimulator 是核心模拟器，统筹所有子系统。

**职责**:
- 管理所有实体（猎人、猎物）
- 协调物理、传感器、能量系统
- 执行每帧更新循环
- 提供性能统计

### 2.2 关键方法

```python
class WorldSimulator:
    def __init__(
        self,
        env_config: EnvConfig = None,
        agent_config: AgentConfig = None,
        use_parallel: bool = True
    ):
        """
        初始化模拟器

        Args:
            env_config: 环境配置（世界大小、能量系统）
            agent_config: 智能体配置（速度、视野）
            use_parallel: 是否启用并行优化
        """

    def initialize(
        self,
        n_hunters: int,
        n_prey: int
    ) -> None:
        """
        初始化实体

        Args:
            n_hunters: 猎人数量
            n_prey: 猎物数量
        """

    def step(self) -> WorldState:
        """
        执行一步模拟

        Returns:
            WorldState: 当前世界状态
                - tick: 当前帧数
                - entities: 所有存活实体列表
                - stats: 性能统计
        """

    def get_stats(self) -> Dict:
        """
        获取统计信息

        Returns:
            {
                'hunters': 猎人数量,
                'preys': 猎物数量,
                'hunter_avg_energy': 猎人平均能量,
                'prey_avg_energy': 猎物平均能量,
                'total_entities': 总实体数
            }
        """
```

### 2.3 更新循环

每次调用 `step()` 时，执行以下流程：

```python
def step(self):
    # 1. 物理更新
    self.physics_engine.update_all(self.entities)

    # 2. 碰撞检测（猎人捕获猎物）
    captures = self.physics_engine.detect_captures(
        hunters, preys
    )

    # 3. 处理捕获（猎人补充能量，猎物死亡）
    for hunter, prey in captures:
        hunter.energy += CAPTURE_ENERGY_GAIN
        prey.alive = False

    # 4. 能量消耗
    self.energy_system.update_all(self.entities)

    # 5. 移除死亡实体
    self.entities = [e for e in self.entities if e.alive]

    # 6. 传感器更新（计算视野内目标）
    self.sensor_system.update_all(self.entities)

    return WorldState(tick=self.tick, entities=self.entities)
```

### 2.4 使用示例

```python
from core import WorldSimulator
from config import EnvConfig, AgentConfig

# 创建模拟器
simulator = WorldSimulator(
    env_config=EnvConfig(),
    agent_config=AgentConfig(),
    use_parallel=True
)

# 初始化实体
simulator.initialize(n_hunters=6, n_prey=18)

# 运行模拟
for i in range(1000):
    world = simulator.step()
    print(f"Tick {world.tick}: {len(world.entities)} entities alive")

    if i % 100 == 0:
        stats = simulator.get_stats()
        print(f"  Hunters: {stats['hunters']}, Preys: {stats['preys']}")
```

---

## 3. PhysicsEngine - 物理引擎

**文件**: `core/physics.py`

### 3.1 功能

负责所有运动物理和碰撞检测。

**职责**:
- 更新实体位置和朝向
- 应用摩擦力
- 边界处理（墙壁反弹）
- 碰撞检测（QuadTree加速）

### 3.2 运动模型

```python
# 每帧更新
entity.angle += entity.angular_velocity * dt
entity.x += entity.speed * cos(entity.angle) * dt
entity.y += entity.speed * sin(entity.angle) * dt

# 摩擦力
entity.speed *= FRICTION_COEFFICIENT
entity.angular_velocity *= ANGULAR_FRICTION
```

### 3.3 边界处理

```python
def handle_boundaries(self, entity: EntityState):
    """处理墙壁碰撞"""
    margin = entity.radius

    # 左右边界
    if entity.x < margin:
        entity.x = margin
        entity.angle = math.pi - entity.angle  # 反弹
    elif entity.x > self.world_width - margin:
        entity.x = self.world_width - margin
        entity.angle = math.pi - entity.angle

    # 上下边界（类似处理）
    ...
```

### 3.4 碰撞检测

使用 **QuadTree** 空间索引加速：

```python
def detect_captures(
    self,
    hunters: List[EntityState],
    preys: List[EntityState]
) -> List[Tuple[EntityState, EntityState]]:
    """
    检测猎人捕获猎物

    Returns:
        [(hunter, prey), ...] 捕获对列表
    """
    # 使用QuadTree加速
    captures = []
    for hunter in hunters:
        # 查询附近猎物（O(log n)）
        nearby_preys = self.quadtree.query_circle(
            hunter.x, hunter.y,
            CAPTURE_RADIUS
        )

        for prey_id in nearby_preys:
            prey = self.get_entity(prey_id)
            if self.is_collision(hunter, prey):
                captures.append((hunter, prey))

    return captures
```

**性能对比**:
- 线性查找: O(n×m) - 猎人数 × 猎物数
- QuadTree: O(n×log m) - 约15倍提升

---

## 4. SensorSystem - 传感器系统

**文件**: `core/sensors.py`

### 4.1 功能

模拟智能体的视野系统。

**职责**:
- 计算视野内目标
- 扇形视野（FOV）+ 视距限制
- 区分猎人/猎物类型
- 并行计算优化

### 4.2 视野模型

```
                视距 (view_distance)
       ◄──────────────────────────►

         ╱                     ╲
        ╱         FOV          ╲
       ╱        (120°)          ╲
      ╱                           ╲
     ●─────────────────────────────
   智能体
```

**参数**（来自 `AgentConfig`）:
- `HUNTER_FOV_DEG`: 猎人视野角度（默认120°）
- `HUNTER_VIEW_DISTANCE`: 猎人视距（默认250像素）
- `PREY_FOV_DEG`: 猎物视野角度（默认150°）
- `PREY_VIEW_DISTANCE`: 猎物视距（默认300像素）

### 4.3 视野检测算法

```python
def is_in_fov(
    self,
    observer: EntityState,
    target: EntityState
) -> bool:
    """检测目标是否在视野内"""
    # 1. 计算距离
    dx = target.x - observer.x
    dy = target.y - observer.y
    distance = math.sqrt(dx**2 + dy**2)

    # 2. 距离检查
    view_distance = (
        self.agent_config.HUNTER_VIEW_DISTANCE
        if observer.type == "hunter"
        else self.agent_config.PREY_VIEW_DISTANCE
    )
    if distance > view_distance:
        return False

    # 3. 角度检查（扇形视野）
    target_angle = math.atan2(dy, dx)
    angle_diff = abs(normalize_angle(target_angle - observer.angle))

    fov = (
        self.agent_config.HUNTER_FOV_DEG
        if observer.type == "hunter"
        else self.agent_config.PREY_FOV_DEG
    )
    fov_rad = math.radians(fov) / 2

    return angle_diff <= fov_rad
```

### 4.4 并行优化

使用 **ParallelRenderer** 批量处理：

```python
# 串行模式：O(n²)
for entity in entities:
    for other in entities:
        if self.is_in_fov(entity, other):
            entity.visible_targets.append(other)

# 并行模式：O(n log n) + 多线程
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(self.compute_fov, entity, entities)
        for entity in entities
    ]
    results = [f.result() for f in futures]
```

---

## 5. EnergySystem - 能量系统

**文件**: `core/energy.py`

### 5.1 功能

管理智能体能量消耗和死亡判定。

**职责**:
- 计算运动能量消耗
- 处理捕获能量补充
- 判定能量耗尽死亡

### 5.2 能量消耗公式

```python
# 运动消耗（每帧）
base_cost = ENERGY_DECAY_RATE  # 基础代谢
movement_cost = entity.speed * SPEED_ENERGY_COST
turn_cost = abs(entity.angular_velocity) * TURN_ENERGY_COST

total_cost = base_cost + movement_cost + turn_cost
entity.energy -= total_cost
```

**参数**（来自 `EnvConfig`）:
- `ENERGY_DECAY_RATE`: 基础代谢（默认0.05/帧）
- `SPEED_ENERGY_COST`: 速度消耗系数（默认0.01）
- `TURN_ENERGY_COST`: 转向消耗系数（默认0.02）

### 5.3 能量补充

```python
# 猎人捕获猎物
if hunter.captures(prey):
    hunter.energy += CAPTURE_ENERGY_GAIN  # +30.0
    prey.alive = False
```

### 5.4 死亡判定

```python
def update(self, entity: EntityState):
    """更新能量状态"""
    # 消耗能量
    entity.energy -= self.calculate_cost(entity)

    # 能量耗尽死亡
    if entity.energy <= 0:
        entity.alive = False
        entity.energy = 0
```

### 5.5 能量平衡设计

**猎人**:
- 初始能量: 100
- 捕获补充: +30
- 平均消耗: ~1.5/帧
- 续航时间: ~67帧（无捕获）

**猎物**:
- 初始能量: 100
- 无补充机制
- 平均消耗: ~1.2/帧
- 续航时间: ~83帧

**生态平衡**:
- 猎人需要每60-70帧捕获1次才能维持
- 猎物数量应为猎人的2-3倍

---

## 6. 数据模型

**文件**: `models/state.py`

### 6.1 EntityState - 实体状态

```python
@dataclass
class EntityState:
    """单个实体的状态"""

    # 标识
    id: str
    type: str  # "hunter" 或 "prey"
    alive: bool

    # 物理状态
    x: float
    y: float
    angle: float  # 朝向角度（弧度）
    speed: float
    angular_velocity: float
    radius: float  # 碰撞半径

    # 能量
    energy: float

    # 传感器
    fov_range: float  # 视距
    fov_angle: float  # 视野角度（弧度）
    visible_hunters: List[str]  # 视野内猎人ID
    visible_preys: List[str]    # 视野内猎物ID

    # 控制（可选）
    target_id: Optional[str] = None
```

### 6.2 WorldState - 世界状态

```python
@dataclass
class WorldState:
    """整个世界的状态"""
    tick: int  # 当前帧数
    entities: List[EntityState]  # 所有实体
    stats: Dict[str, Any]  # 统计信息
```

---

## 7. 使用示例

### 7.1 完整模拟示例

```python
from core import WorldSimulator
from config import EnvConfig, AgentConfig

# 创建配置
env_cfg = EnvConfig()
env_cfg.WINDOW_WIDTH = 1600
env_cfg.WINDOW_HEIGHT = 1000
env_cfg.ENERGY_DECAY_RATE = 0.05

agent_cfg = AgentConfig()
agent_cfg.HUNTER_SPEED_MAX = 50.0
agent_cfg.HUNTER_FOV_DEG = 120.0
agent_cfg.PREY_SPEED_MAX = 45.0

# 创建模拟器
simulator = WorldSimulator(
    env_config=env_cfg,
    agent_config=agent_cfg,
    use_parallel=True
)

# 初始化
simulator.initialize(n_hunters=10, n_prey=30)

# 运行1000步
for step in range(1000):
    world = simulator.step()

    # 打印统计
    if step % 100 == 0:
        stats = simulator.get_stats()
        print(f"Step {step}:")
        print(f"  Hunters: {stats['hunters']}")
        print(f"  Preys: {stats['preys']}")
        print(f"  Avg Hunter Energy: {stats['hunter_avg_energy']:.1f}")
        print(f"  Avg Prey Energy: {stats['prey_avg_energy']:.1f}")
```

### 7.2 自定义物理参数

```python
# 修改运动参数
agent_cfg.HUNTER_SPEED_MAX = 60.0  # 更快的猎人
agent_cfg.PREY_ANGULAR_VELOCITY_MAX = 0.2  # 更灵活的猎物

# 修改视野参数
agent_cfg.HUNTER_VIEW_DISTANCE = 300.0  # 更远的视距
agent_cfg.PREY_FOV_DEG = 180.0  # 更广的视野

# 修改能量参数
env_cfg.CAPTURE_ENERGY_GAIN = 40.0  # 更高的捕获奖励
env_cfg.ENERGY_DECAY_RATE = 0.03  # 更慢的能量消耗
```

### 7.3 访问实体状态

```python
world = simulator.step()

# 遍历所有实体
for entity in world.entities:
    print(f"{entity.type} {entity.id}:")
    print(f"  Position: ({entity.x:.1f}, {entity.y:.1f})")
    print(f"  Energy: {entity.energy:.1f}")
    print(f"  Speed: {entity.speed:.1f}")
    print(f"  Visible hunters: {len(entity.visible_hunters)}")
    print(f"  Visible preys: {len(entity.visible_preys)}")
```

---

## 📚 相关文档

- [配置系统](CONFIGURATION.md) - 详细参数说明
- [并行优化](PARALLEL_OPTIMIZATION.md) - QuadTree性能优化
- [强化学习环境](RL_ENVIRONMENT.md) - 如何在RL中使用核心模块
- [可视化](VISUALIZATION.md) - 前端渲染器如何使用核心模块

---

## 🔧 调试技巧

### 开启调试日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

simulator = WorldSimulator(...)
# 会输出详细的物理更新、碰撞检测日志
```

### 禁用并行模式（调试时）

```python
simulator = WorldSimulator(use_parallel=False)
# 串行模式更容易调试
```

### 检查实体状态

```python
# 在每步后检查异常值
for entity in world.entities:
    assert entity.energy >= 0, f"{entity.id} has negative energy!"
    assert 0 <= entity.x <= env_cfg.WINDOW_WIDTH
    assert 0 <= entity.y <= env_cfg.WINDOW_HEIGHT
```

---

**核心模块是整个项目的基石，理解它们是深入开发的关键！** 🚀
