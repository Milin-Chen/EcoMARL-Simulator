项目说明与技术概览

本项目是一个使用 PyGame 进行实时渲染的“智能生态模拟器”前端，包含数据源（模拟/文件）、世界状态与实体模型、渲染器与交互事件处理。本文档分为两部分：

—

部分一：项目使用的 PyGame 功能

- 初始化与窗口
  - `pygame.init()`：初始化 PyGame。
  - `pygame.display.set_caption(str)`：设置窗口标题。
  - `pygame.display.set_mode((width, height))`：创建主显示 Surface。
  - `pygame.display.flip()`：将绘制结果提交到屏幕（逐帧刷新）。

- 时间与时钟
  - `pygame.time.Clock()`：创建时钟对象，用于控制帧率与获取 delta。
  - `Clock.tick(fps)`：设定最大帧率并返回毫秒间隔；项目中以 60FPS 为目标。
  - `pygame.time.get_ticks()`：获取自启动以来的毫秒数，用于“轻微蠕动”等动态效果。

- 事件系统与输入
  - `pygame.event.get()`：拉取事件队列。
  - 事件类型：`pygame.QUIT`、`pygame.MOUSEBUTTONDOWN`、`pygame.KEYDOWN`。
  - 鼠标：`pygame.mouse.get_pos()` 获取点击坐标（用于实体选取）。
  - 键盘键位：`pygame.K_p`（暂停切换）、`pygame.K_r`（射线显隐）、`pygame.K_d`（调试面板显隐）、
    `pygame.K_LEFTBRACKET` 与 `pygame.K_RIGHTBRACKET`（FOV 范围缩放）、`pygame.K_MINUS` 与 `pygame.K_EQUALS`（射线数量调整）。

- 绘制与图形
  - 基础图元绘制：`pygame.draw.line`、`pygame.draw.circle`、`pygame.draw.ellipse`、`pygame.draw.polygon`。
  - 透明与旋转：`pygame.SRCALPHA` 创建带透明通道的 Surface，`pygame.transform.rotate(surface, angle)` 旋转图形。
  - `Surface.get_rect(center=(x, y))`：以中心坐标获取矩形用于 `blit` 对齐。
  - `Surface.blit(other, rect)`：将图形绘制到主屏幕（主体椭圆与渐隐残影）。

- 字体与文本
  - `pygame.font.SysFont(name, size)`：创建字体对象（项目使用 Menlo）。
  - `Font.render(text, antialias, color)`：渲染文本到 Surface（调试面板）。

—

部分二：整体逻辑与架构说明（含传入参数规范）

模块结构

- `src/app.py`：应用入口与主循环
  - 选择数据源（`MockSource` 或 `FileJSONSource`）。
  - 每帧流程：`source.poll()` → `renderer.handle_events(world)` → `renderer.draw_world(world)` → `pygame.display.flip()` → `renderer.tick()`。

- `src/datasource.py`：数据源与世界更新
  - 抽象类 `DataSource`：定义 `poll() -> Optional[WorldState]`。
  - `MockSource(n_hunters: int = 6, n_prey: int = 18)`：随机生成并在前端本地更新实体运动学、能量与繁殖冷却，计算视线射线；提供 `_spawn_child(...)` 与 `force_breed(entity_id)`。
  - `FileJSONSource(path: str = "runtime/world.json")`：从 JSON 文件读取 `WorldState`，支持后端联调。
  - （占位）`WebSocketSource(url: str = "ws://localhost:8765")`：为未来的实时后端预留接口形状。

- `src/render.py`：渲染与交互
  - 类 `PygameRenderer`：窗口初始化、帧绘制、事件处理、调试面板。
  - 视觉要素：背景网格、实体圆形与速度方向拉伸椭圆、渐隐残影、选中高亮、能量条、FOV 扇形、双眼与 FOV 限制瞳孔。
  - 交互：点击选中实体（使用平滑插值坐标与新生体放大命中半径），键盘控制面板与 FOV/射线。

- `src/models.py`：数据模型
  - `WorldState`、`EntityState`、`RayHit` 及占位 `NeuralNetSpec`，用于承载渲染所需的全部状态。

- `src/config.py`：参数与常量
  - 画布大小、颜色、FOV/射线默认值、实体半径与动画参数、调试面板样式、分裂成长与繁殖冷却等配置。

数据流与更新规则

- 每帧从数据源获取世界状态；若 `poll()` 返回 `None`，沿用上一帧以保证平滑显示。
- `MockSource._update_motion(dt)`：能量衰减、年龄累计、繁殖冷却递减、分裂成长（`spawn_progress`）、边界反弹；
  - 捕食者在能量为 0 时移除；猎物能量为 0 时原地不动但仍渲染与可选中。
  - 随机“捕食”事件：捕食者吃到猎物后获得能量并短暂消化。
  - 分裂逻辑：按类型与能量/年龄/冷却控制；`_spawn_child(...)` 继承父体属性并设置 `breed_cd` 与 `spawn_progress`。
- 渲染器根据最近实体映射控制“目光方向”，并在调试面板中显示“存活时间、Gen、Children、FOV、Digestion”等。

传入参数的规范（后端/文件输入）

- 顶层结构（`WorldState.from_dict(payload)`）：
  - `tick: int`（必选）：世界迭代计数。
  - `entities: List[Entity]`（必选）：实体列表。

- `Entity` 对象字段（均按 `EntityState` 转换；未提供时按默认）：
  - `id: str`（必选）
  - `type: Literal["hunter","prey"]`（必选）
  - `x: float`、`y: float`（必选）：位置（像素）。
  - `angle: float`（必选）：朝向（弧度）。
  - `speed: float`、`angular_velocity: float`：线速度（px/s）与角速度（rad/s）。
  - `radius: float`：渲染半径。
  - `energy: float`、`digestion: float`、`age: float`、`generation: int`、`offspring_count: int`
  - FOV：`fov_deg: float`、`fov_range: float`
  - `rays: List[RayHit]`：每项含 `angle: float`、`distance: float`、可选 `hit_type: Literal["hunter","prey"]`、`hit_id: str`
  - 前端扩展（可选）：
    - `split_energy: float`：分裂能量阈值（默认 120）。
    - `target_id: str | null`
    - `iteration: int`：用于显示的迭代计数（默认为 `tick`）。
    - `breed_cd: float`：繁殖冷却（秒）。
    - `spawn_progress: float`：分裂成长进度（0..1）。
    - `should_persist: bool`、`lifespan: float | null`、`saved: bool`

- 文件数据源（`FileJSONSource`）
  - 路径参数：`path: str = "runtime/world.json"`
  - 读取失败或未更新时返回 `None`，渲染器沿用上一帧。

- 随机数据源（`MockSource`）构造参数
  - `n_hunters: int`：初始捕食者数量（默认 6）。
  - `n_prey: int`：初始猎物数量（默认 18）。
  - 运行时外部触发接口：`force_breed(entity_id: str) -> Optional[EntityState]`（尊重 `MAX_ENTITIES` 与类型冷却）。

约定与注意事项

- 坐标系：左上角为原点 `(0,0)`，右下为 `(WINDOW_WIDTH, WINDOW_HEIGHT)`。
- 时间单位：秒；角度单位：弧度；速度单位：像素/秒。
- 渲染透明度与旋转需要 `SRCALPHA` Surface 与 `transform.rotate`，注意先构造再旋转以避免锯齿。
- 调试面板渲染使用系统字体 Menlo，若缺失可替换为等宽字体以保持布局。