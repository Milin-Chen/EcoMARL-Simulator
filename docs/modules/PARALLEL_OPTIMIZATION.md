# 性能优化详解

**Parallel Optimization - QuadTree与多线程加速**

---

## 1. QuadTree空间索引

### 1.1 原理

QuadTree将2D空间递归分割成4个象限，实现高效的空间查询。

```
┌─────────────────┐
│     NW │ NE     │
│   ●    │    ●   │
├────────┼────────┤
│  ●  SW │ SE   ● │
│        │     ●  │
└─────────────────┘
```

### 1.2 性能对比

| 实体数 | 线性查找O(n²) | QuadTree O(n log n) | 提升 |
|-------|--------------|---------------------|------|
| 20    | 400次        | ~86次               | 5x   |
| 100   | 10,000次     | ~664次              | 15x  |
| 200   | 40,000次     | ~1,529次            | 26x  |

### 1.3 使用场景

- **碰撞检测**: 猎人捕获猎物
- **视野查询**: 查找视野内实体
- **最近邻查找**: 前端渲染目标查找

### 1.4 实现

**文件**: `parallel/quadtree.py`

```python
from parallel import QuadTree

# 创建QuadTree
qt = QuadTree(width=1600, height=1000)

# 插入实体
for entity in entities:
    qt.insert(entity.id, entity.x, entity.y, entity.radius)

# 查询附近实体
nearby_ids = qt.query_circle(x=800, y=500, radius=100)

# 清空并重建（每帧）
qt.clear()
```

---

## 2. 并行加速

### 2.1 ParallelRenderer

**文件**: `parallel/renderer.py`

```python
from parallel import ParallelRenderer

renderer = ParallelRenderer(
    world_width=1600,
    world_height=1000,
    use_quadtree=True
)

# 批量碰撞检测
collisions = renderer.detect_collisions_parallel(
    hunters=hunter_list,
    preys=prey_list
)

# 批量视野计算
fov_results = renderer.compute_fov_parallel(
    entities=all_entities
)
```

### 2.2 多线程池

使用 `ThreadPoolExecutor` 并行处理：

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(process_entity, entity)
        for entity in entities
    ]
    results = [f.result() for f in futures]
```

---

## 3. 前后端统一优化

### 3.1 后端优化

**位置**: `core/world.py`

```python
simulator = WorldSimulator(use_parallel=True)
# 自动启用ParallelRenderer加速碰撞和传感器
```

### 3.2 前端优化

**位置**: `frontend/pygame_renderer.py`

```python
renderer = PygameRenderer(use_parallel=True)
# 自动启用QuadTree加速最近邻查找
```

### 3.3 启用/禁用

```bash
# 启用（默认）
python main.py

# 禁用（调试用）
python main.py serial
```

---

## 📚 相关文档

- [核心模块](CORE_MODULES.md) - 底层物理引擎
- [可视化](VISUALIZATION.md) - 前端渲染

---

**性能优化，流畅体验！** ⚡
