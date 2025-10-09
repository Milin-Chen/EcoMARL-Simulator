技术拆解与重构参考

本文面向后续重构与参数对接，逐文件、逐模块解释当前实现，并系统性介绍 PyGame 的核心概念与在本项目中的使用方式。

—

一、PyGame 概念与项目中的用法

1) 核心概念
- Surface：PyGame 的二维画布对象，所有绘制都在 `Surface` 上进行；主屏幕由 `pygame.display.set_mode(...)` 返回的 `Surface` 表示。
- Rect：矩形区域描述体，常用于定位与碰撞；许多 API 接受 `Rect` 控制位置与大小，如 `Surface.get_rect(center=(x,y))`。
- Event Loop：事件循环，拉取事件队列（鼠标、键盘、窗口事件等），并进行响应；使用 `pygame.event.get()`。
- Clock：帧率与时间控制器，`Clock.tick(fps)` 限制最大帧率并返回时间步；用于动画与物理更新。
- Blit：将一个 `Surface` 绘制到另一个 `Surface`，如将旋转后的椭圆绘制到屏幕：`screen.blit(rotated_surface, rect)`。
- Transform：图像变换，如 `pygame.transform.rotate(surface, angle)` 进行旋转；搭配 `SRCALPHA` 支持透明通道。
- 颜色与透明：颜色为 RGB 三元组 `(r,g,b)`；支持含透明度的颜色 `(r,g,b,a)`；使用 `pygame.SRCALPHA` 创建有 Alpha 通道的 `Surface`。
- 字体渲染：`pygame.font.SysFont(name, size)` 创建字体；`font.render(text, antialias, color)` 将文本渲染成 `Surface`。
- 坐标系与单位：屏幕左上角是原点 `(0,0)`，x 右增、y 下增；角度以弧度为主（部分显示转为度），速度为像素/秒。

2) 项目内对应用法（示意）
- 初始化窗口：`pygame.init()` → `screen = pygame.display.set_mode((W,H))` → `pygame.display.set_caption(...)`。
- 时钟与帧：`clock = pygame.time.Clock()`；`clock.tick(60)` 控制 60FPS；`pygame.time.get_ticks()/1000.0` 用于生成蠕动时间参数。
- 事件：`for event in pygame.event.get(): ...`，处理 `QUIT/MOUSEBUTTONDOWN/KEYDOWN` 等。
- 绘制：
  - `pygame.draw.line/circle/ellipse/polygon` 画基础图形；
  - 先在带 `SRCALPHA` 的临时 `Surface` 上画椭圆，再 `transform.rotate`，最后 `blit` 到屏幕，避免直接旋转导致锯齿；
  - 调试面板使用 `font.render` 渲染文字。
- 提交显示：每帧末尾 `pygame.display.flip()`。

—

二、逐文件核心逻辑与函数详解

A) `src/app.py`
- 作用：应用入口与主循环。
- 关键流程：
  - 选择数据源：`source = MockSource(...)` 或 `FileJSONSource(path=...)`（可按需切换）。
  - 初始化渲染器：`renderer = PygameRenderer()`（内部创建窗口与时钟）。
  - 主循环：
    - `world = source.poll() or last_frame`：无新帧时沿用上一帧（保证平滑渲染）。
    - `renderer.handle_events(world)`：处理输入事件（点击选中、面板切换等）。
    - `renderer.draw_world(world)`：绘制所有实体与调试面板。
    - `pygame.display.flip()`：提交显示。
    - `renderer.tick()`：控制帧率（60FPS）。

B) `src/datasource.py`
- 作用：生成或加载世界状态（实体位置、能量、分裂、射线等）。
- 核心类与函数：
  - `class DataSource`：抽象基类，定义 `poll() -> Optional[WorldState]` 接口。
  - `class MockSource(DataSource)`：本地模拟源，前端独立运行演示用。
    - `__init__(n_hunters, n_prey)`：
      - 随机生成初始实体（位置、速度、角度、FOV 等），类型为 `hunter/prey`；
      - 使用 `config.DEFAULT_FOV_*`、`DEFAULT_RADIUS` 等常量。返回 `EntityState`。
    - `_spawn_child(parent, etype=None) -> EntityState`：
      - 依据父体属性生成子体（位置沿父体角度方向偏移 `off`，速度与角速度小范围扰动）；
      - 子体：`generation = parent.generation + 1`、能源 `= parent.energy * 0.5`、`spawn_progress=0.0`（由小到大平滑长成）；
      - 按类型设置冷却：`breed_cd = BREED_CD_PREY/HUNTER`；父体进入相同类型冷却并能量减半、`offspring_count+1`；
      - 超过 `config.MAX_ENTITIES` 时不追加。
    - `force_breed(entity_id) -> Optional[EntityState]`：
      - 外部触发分裂接口；查找父体并调用 `_spawn_child`（尊重最大实体数与类型冷却）。
    - `_update_motion(dt)`：
      - 能量衰减（猎人更快）、年龄累加、`breed_cd` 递减、`spawn_progress` 增长（子体半径比例向 1.0 收敛）；
      - 运动学更新：当猎物能量为 0 时停止移动但仍渲染；猎人能量为 0 时移除；
      - 边界反弹：触边则反向并加 90°；
      - 简易“捕食事件”：猎人靠近猎物获得能量并进入 `digestion`（消化倒计时），期间不再捕食；
      - 分裂触发：猎人能量与冷却满足且随机触发；猎物满足年龄或能量阈值且随机触发；
      - 移除被吃掉的猎物（`eaten_ids`）。
    - `_compute_rays(e) -> List[RayHit]`：
      - 在 `e.angle ± fov_deg/2` 范围均匀发射 `DEFAULT_RAY_COUNT` 条射线；
      - 与其他实体的圆形近似做几何相交近似，计算沿射线方向的最近距离；
      - 返回 `RayHit(angle, distance, hit_type, hit_id)`；若无命中则距离为 `fov_range`。
    - `poll()`：
      - `tick += 1`、`dt = 1/60`，调用 `_update_motion(dt)`；
      - 每个实体更新 `rays` 与 `iteration`；
      - 返回 `WorldState(tick, entities)`。
  - `class FileJSONSource(DataSource)`：从 JSON 文件读取；
    - `poll()`：基于文件 `mtime` 增量读取，调用 `WorldState.from_dict(payload)`；读取失败返回 `None`。
  - `class WebSocketSource(DataSource)`：占位（未来可接入实时后端）。

C) `src/models.py`
- 作用：数据结构定义。
- 核心数据：
  - `EntityType = Literal["hunter","prey"]`。
  - `RayHit(angle, distance, hit_type?, hit_id?)`：视线射线的命中结果。
  - `NeuralNetSpec(layers, has_bias, mutation)`：前端展示占位；可承载后端网络结构摘要。
  - `EntityState`：实体的全部状态（位置、速度、能量、年龄、代数、子代数、FOV、rays、扩展字段如 `split_energy/breed_cd/spawn_progress` 等）。
  - `WorldState(tick, entities)` 与 `from_dict(payload)`：将外部 JSON 解析为内部对象（字段缺省自动填默认值）。

D) `src/render.py`
- 作用：渲染循环与交互。
- 主要成员：
  - `PygameRenderer.__init__`：初始化 PyGame、创建屏幕 `Surface`、时钟、字体；定义交互开关（暂停、射线可见、调试面板等），以及平滑插值缓存 `_smooth` 与历史位置 `_prev_draw`。
  - `_lerp_state(e)`：对 `(x,y,angle)` 做指数插值平滑；叠加轻微“蠕动”噪声，让移动更自然；更新 `_smooth`。
  - `_draw_grid()`：填充背景色 `config.BG_COLOR` 并绘制网格线。
  - `_draw_stretched_body(x,y,move_a,r,color,speed)`：按速度方向将圆拉伸为椭圆，并沿反方向绘制多层渐隐残影（透明度逐层降低），最后旋转后 `blit` 到屏幕。
  - `_draw_entity(e, gaze_dir, gaze_dist)`：
    - 使用 `_lerp_state` 获取平滑坐标与角度；
    - 半径按 `spawn_progress` 从小到大缩放；
    - 绘制拉伸主体 + 主圆 + 朝向线；
    - 选中时绘制能量条与 FOV 扇形；
    - 绘制双眼与瞳孔：瞳孔大小受最近实体距离影响；瞳孔偏移角度被限制在实体的 FOV 范围内；
    - 选中时绘制射线与外圈高亮；
    - 更新 `_prev_draw`。
  - `_pick_entity(world, pos)`：
    - 使用与渲染一致的平滑位置进行命中检测；
    - 对新生体（`spawn_progress<0.7`）适度增大点击半径；
    - 选中已死亡猎人会被过滤；将最近的命中实体 id 记录到 `selected_id`。
  - `_draw_debug_panel(world)`：
    - 左上角透明面板，显示 Tick/Entities 等概览；
    - 显示选中实体的概要：能量、速度、存活时间（`mm:ss.ss`）、Gen、Children、FOV、Digestion/Target（猎人）等。
  - `handle_events(world)`：
    - 鼠标左键：选中实体；
    - 键盘：`p/r/d` 切换状态、`[`/`]` 调整 FOV 范围、`-`/`=` 调整射线数量；
    - `QUIT` 事件退出主循环。
  - `draw_world(world)`：
    - 根据 `energy<=0` 的猎人过滤 `alive` 列表；
    - 为每个实体找到最近邻（用于瞳孔朝向与距离感）；
    - 绘制实体与调试面板。
  - `tick()`：`clock.tick(60)` 控制帧率。

E) `src/config.py`
- 作用：常量与样式配置。
- 关键项：窗口大小与颜色、FOV 默认值与射线数量、实体半径与动画参数（拉伸、残影、蠕动）、调试面板样式、分裂成长（`SPAWN_*`）与繁殖冷却（`BREED_CD_*`）、实体上限（`MAX_ENTITIES`）。

—

三、外部输入与参数对接规范（摘要）

1) 顶层结构 `WorldState`
- `tick: int` 必填；
- `entities: List[EntityState]` 必填。

2) `EntityState` 字段（常用）
- 基本：`id, type, x, y, angle, speed, angular_velocity, radius`；
- 状态：`energy, digestion, age, generation, offspring_count`；
- 视野与射线：`fov_deg, fov_range, rays: List[RayHit(angle,distance,hit_type?,hit_id?)]`；
- 前端扩展：`split_energy, target_id?, iteration, breed_cd, spawn_progress, should_persist, lifespan?, saved`。

3) 数据源
- `FileJSONSource(path)`：从文件读取，失败或未更新返回 `None`；
- `MockSource(n_hunters, n_prey)`：本地生成；可通过 `force_breed(entity_id)` 立即分裂（受 `MAX_ENTITIES` 与冷却限制）。

—

四、重构关注点与建议

- 渲染耦合点：
  - `_draw_entity` 集成了缩放（`spawn_progress`）、瞳孔与 FOV、能量条与射线绘制，重构时可拆分为多个小绘制函数（主体、装饰、HUD）。
  - 命中检测与平滑位置：保持 `_lerp_state` 与 `_pick_entity` 使用同一坐标源，避免点击与绘制错位。

- 数据更新策略：
  - 将“显示相关”的字段（`spawn_progress/iteration`）与“物理/生存相关”字段（`energy/age`）区分，便于后端接入时只负责物理；
  - `_compute_rays` 可由后端替换为更精确的几何/物理求交；前端仅做占位。

- 参数配置化：
  - `config` 中的阈值（能量、冷却、分裂成长速率等）建议集中成“实验参数”小节，便于不同仿真环境切换；
  - 面板、瞳孔比例等 UI 相关参数与物理参数分离。

- 性能与可读性：
  - 椭圆残影绘制可按 `speed` 动态减少层数以节流；
  - `_lerp_state` 的蠕动噪声使用 `get_ticks()`，在需要确定性回放时可改为基于 `tick` 的伪随机。