智能生态模拟器前端（PyGame）

项目简介
- 使用 Python + PyGame 在连续 2D 空间渲染生态系统：捕食者（红色）与猎物（绿色）。
- 支持平滑运动、FOV 射线、眼睛/瞳孔变化，以及事件驱动的吞咽带与喂食脉冲、繁殖与成长特效。
- 与后端通过 JSON 帧对接（文件轮询或 WebSocket 占位），也可使用内置 Mock 源快速演示。

快速开始
- 一键运行（推荐）：
  - `bash scripts/run.sh`
- 手动运行：
  - `python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt`
  - `./.venv/bin/python src/app.py`

目录结构
- `src/app.py`：应用入口，选择数据源并启动渲染循环。
- `src/render.py`：渲染器，实现事件驱动效果、调试面板、交互与绘制。
- `src/models.py`：世界与实体数据结构定义，含事件 `Event` 与解析 `WorldState.from_dict`。
- `src/datasource.py`：数据源抽象，内置 `MockSource` 与 `FileJSONSource`。
- `src/config.py`：全局渲染配置（颜色、尺寸、FOV、动画参数等）。
- `scripts/run.sh`：自动创建虚拟环境、安装依赖并运行。
- `requirements.txt`：依赖列表。

数据源与事件对接
- 模式一：文件 JSON（推荐起步）
  - 后端每帧写入 `runtime/world.json`；前端使用 `FileJSONSource` 轮询读取并渲染。
- 模式二：WebSocket（占位）
  - 预留地址：`ws://localhost:8765`（可按需实现与替换）。

顶层 JSON 结构（每帧）
```json
{
  "tick": 123,
  "entities": [
    {
      "id": "H-1",
      "type": "hunter",  
      "x": 100.0, "y": 200.0,
      "angle": 1.57,
      "speed": 32.0,
      "angular_velocity": 0.1,
      "radius": 8.0,
      "energy": 85.0,
      "digestion": 0.0,
      "age": 12.3,
      "generation": 2,
      "offspring_count": 1,
      "fov_deg": 40.0,
      "fov_range": 220.0,
      "rays": [
        {"angle": 1.2, "distance": 80.0, "hit_type": "prey", "hit_id": "P-7"}
      ],
      "spawn_progress": 1.0
    }
  ],
  "events": [
    {"type":"predation","actor_id":"H-17","target_id":"P-201","energy_gain":18.5},
    {"type":"breed","parent_id":"H-8","child":{"id":"H-8-1","type":"hunter","x":420,"y":260,"angle":0.1,"radius":9}}
  ],
  "counters": {
    "predations": 12,
    "births": 3,
    "predator_kills": {"H-17": 4, "H-8": 2}
  }
}
```

事件字段说明
- `predation`（捕食）：
  - `type`: 固定为 `"predation"`
  - `actor_id`: 捕食者 id（必填，前端用于触发吞咽/脉冲与击杀计数）
  - `target_id`: 猎物 id（推荐传，便于调试与记录）
  - `energy_gain`: 本次进食获得的能量（浮点，可选；影响吞咽带与喂食脉冲幅度）
- `breed`（繁殖）：
  - `type`: 固定为 `"breed"`
  - `parent_id`: 父体 id（可选；也可在 `child.parent_id` 中提供，前端都能识别）
  - `child`: 子体最小描述（建议包含 `id/type/x/y/angle/radius`；前端将从 0→1 平滑成长）

计数器字段说明（可选）
- `predations`: 总捕食次数。
- `births`: 总出生数。
- `predator_kills`: 每个捕食者的击杀数映射（前端调试面板展示 `Kills`）。

兜底统计（前端侧）
- 击杀数兜底：当前端未从 `counters.predator_kills` 获取到权威值时，前端按 `predation` 事件为每个捕食者递增本地击杀计数。
- 子代/父代兜底：
  - 在收到 `breed` 事件时，若能识别到父体（从 `parent_id` 或 `child.parent_id`），前端将本地累计该父体的 `offspring` 次数。
  - 若事件包含 `child.id` 且能在本帧实体中找到父体，前端将本地推导子体的 `generation = parent.generation + 1`。
  - 展示规则：调试面板使用“实体字段与兜底值的最大值”进行显示，例如 `Children = max(entity.offspring_count, fallback)`，`Gen = max(entity.generation, fallback)`。
  - 作用范围：仅用于“本地可视化兜底”。权威统计仍建议由后端计算并随实体或 `counters` 一并传输。

渲染与交互
- 事件驱动效果：
  - 捕食：吞咽带沿体长移动、喂食脉冲增强；击杀计数递增。
  - 繁殖：子体平滑成长；若未提供 `spawn_progress`，前端自动推进。
- 调试面板（左上角）：
  - 选中实体显示：能量、速度、分裂能量、视野参数、目标追踪、`Kills`（击杀统计），以及 `Gen/Children`（代数/子代数）。其中 `Gen/Children` 将采用实体字段与前端兜底值的最大值展示。
- 快捷键：
  - `p` 暂停/继续
  - `r` 显示/隐藏射线
  - `d` 显示/隐藏调试面板
  - `[` / `]` 调整 FOV 距离缩放
  - `-` / `=` 减少/增加每实体射线数量

数据源切换
- 默认使用 `MockSource`（在 `src/app.py`）：
```python
# source = FileJSONSource(path="runtime/world.json")
source = MockSource(n_hunters=8, n_prey=24)
```
- 改为使用文件 JSON 并启用事件驱动：
```python
source = FileJSONSource(path="runtime/world.json")
# source = MockSource(n_hunters=8, n_prey=24)
```
- 请在 `runtime/world.json` 提供符合上述结构的帧（含可选 `events` 与 `counters`）。

关键代码点
- `src/models.py`
  - `Event`：事件数据结构，支持 `predation` / `breed` 等类型。
  - `WorldState.from_dict`：解析 `entities`、`events` 与可选 `counters`。
- `src/render.py`
  - `PygameRenderer.draw_world`：调用 `_process_events` 并推进成长覆盖；绘制实体与面板。
  - `_process_events(world)`：根据事件触发吞咽/脉冲、记录击杀与出生、合并后端计数。
  - `_draw_entity(...)`：事件优先的成长缩放；捕食时的吞咽带与喂食脉冲效果。
- `src/datasource.py`
  - `FileJSONSource.poll`：读取 JSON 并使用 `WorldState.from_dict` 解析。
  - `MockSource`：演示用数据源（不含事件）。

常见问题（FAQ）
- 看不到吞咽带/喂食脉冲？
  - Mock 源没有事件；请切换到 `FileJSONSource` 并在 JSON 帧中提供 `predation` 事件。
- 子体如何平滑成长？
  - 后端可传 `spawn_progress`；若无，前端在收到 `breed` 事件后会按 `SPAWN_GROW_RATE` 自动增长。
- 击杀数如何统计？
  - 前端根据 `predation` 事件按捕食者累计；若后端提供 `counters.predator_kills`，前端直接读取覆盖。
- 代数/子代数如何统计？
  - 权威口径建议由后端在实体字段中提供 `generation` 与 `offspring_count`。
  - 若后端暂未提供，前端会基于 `breed` 事件进行“本地兜底展示”：从父体 ID 累加子代次数，并在识别到父体与子体时推导 `generation = parent.generation + 1`；展示取实体值与兜底值的最大值。
  - 注意：兜底统计在刷新或切换数据源后会重置；如需长期一致性与多前端一致，请使用后端统计。


```

兼容性与环境
- 已在 macOS 上使用 Python 3.13 + `pygame-ce 2.5.2` 验证运行。
- Windows 用户可用 PowerShell 执行：
```powershell
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt
./.venv/Scripts/python src/app.py
```

实体字段可选性与配置默认
- 后端可以省略以下实体字段，前端将使用安全默认或配置值：
  - `angle`：可不传，默认 `0.0`（朝向 +x 轴）。
  - `radius`：可不传，默认 `config.DEFAULT_RADIUS`。
  - `fov_deg` / `fov_range`：可不传；当 `config.USE_ENTITY_FOV=True` 时：
    - 若实体未提供则按类型使用默认：`HUNTER_FOV_DEG/HUNTER_FOV_RANGE`、`PREY_FOV_DEG/PREY_FOV_RANGE`。
    - 若实体提供则优先使用实体值。
  - `rays`：可不传；前端可基于几何近似计算用于调试与可视化。
  - `spawn_progress`：可不传；若收到 `breed` 事件，前端将从 0→1 平滑成长（速率 `config.SPAWN_GROW_RATE`）。
  - `breed_cd`：可不传；前端调试时可使用类型默认 `config.BREED_CD_PREY` / `config.BREED_CD_HUNTER`。
  - `iteration`/`target_id`/`nn` 等：可不传；前端会根据 `tick`/最近目标/占位网络进行展示与调试。

事件与计数的可选性
- `events`：可选；为空时不会触发吞咽与成长事件，界面正常渲染。
- `counters`：可选；前端可自行根据 `events` 累计总捕食次数与每捕食者击杀数；
  - 若提供 `counters.predator_kills`（形如 `{"H-17":4}`），前端直接读取并覆盖展示。

按场景的配置参数指南（config.py）
- 视觉/感知配置：
  - `HUNTER_FOV_DEG/HUNTER_FOV_RANGE`、`PREY_FOV_DEG/PREY_FOV_RANGE`：不同类型默认 FOV；适合“视觉对比”场景。
  - `USE_ENTITY_FOV`：为真时优先用实体的 FOV；适合后端已计算个体化视角的场景。
  - `DEFAULT_RAY_COUNT`：射线数量基准；调大提升精度，代价是性能。
- 捕食动画与触发：
  - `FEED_ENERGY_DELTA_THRESHOLD`：能量增量阈值，超过触发吞咽带与脉冲；低阈值适合“高频小口”场景。
  - `FEED_PULSE_GAIN/FEED_PULSE_DECAY/FEED_PULSE_MAX` 与 `FEED_RING_*`：控制脉冲幅度、衰减与环尺寸。
  - `SWALLOW_*`：控制吞咽带颜色、透明度、速度与尺寸比例；“快速吞咽”可调高 `SWALLOW_SPEED` 并略降 `SWALLOW_DECAY`。
- 繁殖与成长：
  - `BREED_CD_PREY/BREED_CD_HUNTER`：不同类型繁殖冷却；“群体爆发增长”可降低冷却。
  - `SPAWN_MIN_SCALE/SPAWN_GROW_RATE`：子体初始尺寸与成长速率；“幼体展示”可降低最小比例并增加成长时间。
- 运动与形变：
  - `SMOOTH_LERP`：插值速率；更平滑的轨迹可略降，代价是响应延迟。
  - `STRETCH_*` 与 `SOFT_BODY_*`：拉伸残影与果冻形变参数；“高速水滴形态”可提高 `STRETCH_SPEED_SCALE`、`SOFT_BODY_HEAD_COMPRESS`。
  - `DROPLET_*`：水滴尾迹形态与透明度；开启 `DROPLET_TRAIL_ENABLED` 可获得更强的速度感（有性能开销）。

相机跟随与缩放（全局）
- 功能概述：当在画面中点击选中一个实体时，前端将以该实体为中心进行“全局相机跟随与缩放”，使其运动更易观察；取消选中或实体不再存在时，相机平滑恢复到 1.0 缩放。
- 相关配置（`src/config.py`）：
  - `CAMERA_ZOOM_SELECTED`：选中实体时的全局缩放倍数（默认 `1.8`）。可按需要调小到 `1.3~1.5` 以提升性能。
  - `CAMERA_LERP`：相机缩放与居中平滑插值系数（默认 `0.18`）。数值越大，跟随越快但可能更突兀；越小越平滑。
- 行为说明：
  - 相机以选中实体的平滑位置为视口中心，计算一个矩形子视图并缩放到窗口大小展示。
  - 传感器条与调试面板等 UI 作为屏幕空间元素，不随相机缩放，以保证可读性。
- 性能注意：
  - 全局缩放会对离屏图层进行裁剪与缩放，缩放倍数越高开销越大；建议优先调整 `CAMERA_ZOOM_SELECTED` 与 `CAMERA_LERP` 取得平衡。
  - 如需进一步优化，可将缩放倍数降低、减少软体节点数（`SOFT_BODY_NODES`），或改为坐标级相机（在渲染阶段直接缩放坐标与半径，避免图层像素缩放）。

软体运动调试步骤
- 前置设置：启动程序后按 `d` 打开调试面板；按 `r` 显示/隐藏射线，按 `n` 切换传感器条以辅助观察方向与命中。
- 观察基线：在默认参数下记录形变与响应感觉（抖动、回弹、收敛速度）。
- 参数焦点：
  - 细腻度与稳定性：`SOFT_BODY_NODES`（节点数）、`SOFT_BODY_SPRING_K`（弹簧强度）、`SOFT_BODY_DAMPING`（阻尼）。节点越多越细腻，阻尼越大越稳。
  - 速度耦合形变：`SOFT_BODY_HEAD_COMPRESS` 与 `SOFT_BODY_TAIL_ELONGATE` 控制“头压缩/尾拉伸”。增大它们可增强“水滴”效果。
  - 微抖动：`SOFT_BODY_WOBBLE_BASE` 与 `SOFT_BODY_WOBBLE_ANG` 控制基础抖动与角速度增益；抖动过大时提高阻尼以抑制震荡。
  - 平滑插值：`SMOOTH_LERP` 决定运动平滑度；增大更平滑但响应更慢。
- 推荐流程：每次只调整一个参数→观察 5–10 秒→记录结果与可接受范围→必要时回退。

吞咽与繁殖效果调试步骤
- 事件准备：如需精准复现，请切换到 `FileJSONSource` 并在 `runtime/world.json` 中提供事件；也可在 `MockSource` 下仅观察前端视觉效果。
- 捕食（吞咽带与能量脉冲）：
  - 触发条件：能量增量超过 `FEED_ENERGY_DELTA_THRESHOLD` 或收到 `predation` 事件。
  - 视觉参数：`SWALLOW_*` 控制吞咽带的长度、宽高与速度；`FEED_PULSE_*` 控制能量脉冲环的幅度、衰减与上限。
  - 操作步骤：
    1) 按 `d` 开启调试面板，关注 `Kills` 与能量变化；
    2) 缓慢调高 `SWALLOW_SPEED` 与适度降低 `SWALLOW_DECAY` 观察带状高亮的长度与消退；
    3) 调整 `FEED_PULSE_GAIN/DECAY/MAX`，确认脉冲环边缘清晰且能在 1–2 秒内淡出；
    4) 选中捕食者，确认摄像机跟随与缩放下效果仍自然。
  - 事件示例（最小片段）：
```json
{
  "tick": 123,
  "entities": [
    {"id":"H-1","type":"hunter","x":300,"y":200,"angle":0.5,"energy":12.0},
    {"id":"P-9","type":"prey","x":320,"y":210,"angle":-0.3,"energy":4.0}
  ],
  "events": [
    {"type":"predation","predator_id":"H-1","prey_id":"P-9","energy_delta":1.2}
  ]
}
```
- 繁殖（平滑成长）：
  - 触发条件：收到 `breed` 事件。
  - 视觉参数：`SPAWN_MIN_SCALE`（子体初始比例）与 `SPAWN_GROW_RATE`（成长速率）。
  - 操作步骤：
    1) 提供 `breed` 事件，按 `d` 开启调试面板，观察 `Children` 与 `Gen` 展示；
    2) 调整 `SPAWN_MIN_SCALE` 以控制起始大小，适度调 `SPAWN_GROW_RATE` 让成长在 0.8–2.0 秒间完成；
    3) 确认子体在成长过程中具有连贯位置与角度（来自后端或前端平滑）。
  - 事件示例（最小片段）：
```json
{
  "tick": 456,
  "entities": [
    {"id":"P-2","type":"prey","x":420,"y":300,"angle":0.1,"energy":5.0}
  ],
  "events": [
    {"type":"breed","parent_id":"P-2","child":{"id":"P-2-1","type":"prey","x":428,"y":304,"angle":0.1}}
  ]
}
```

验证清单
- 选中实体后相机居中与缩放正常；UI 不随缩放失真。
- 捕食时吞咽带沿运动方向移动，能量脉冲环平滑淡出；`Kills` 统计符合预期。
- 繁殖时子体从 `SPAWN_MIN_SCALE` 平滑增长到目标半径，父子代数/子代计数展示正确（若后端未提供则使用前端兜底）。