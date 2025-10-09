智能生态模拟器前端（PyGame）

概述
- 使用 Python + PyGame 在连续 2D 空间渲染生态系统：猎人（红色）与猎物（绿色）。
- 支持平滑运动、FOV 射线、眼睛/瞳孔变化、点击调试面板等可视化。
- 预留与后端物理引擎（pymunk）交互的接口：文件 JSON / WebSocket（占位）。

运行
1. 安装依赖：
   - 建议创建虚拟环境：`python3 -m venv .venv && source .venv/bin/activate`
   - 安装：`pip install -r requirements.txt`
2. 启动：`.venv/bin/python src/app.py`

目录结构
- `src/app.py`：程序入口与主循环。
- `src/render.py`：渲染器，绘制实体、FOV、眼睛与调试面板。
- `src/models.py`：世界与实体数据结构，适配来自 pymunk 的参数。
- `src/datasource.py`：数据源抽象与 Mock/JSON/WebSocket 占位实现。
- `src/config.py`：全局配置（颜色、尺寸、FOV 等）。
- `requirements.txt`：依赖列表。

与后端（pymunk）对接
- 文件 JSON 模式（推荐起步）：后端每帧写入 `runtime/world.json`，前端轮询读取。
- 预留的 WebSocket 模式：`ws://localhost:8765`（可自行更改）。
- 统一数据格式（每帧）：
  ```json
  {
    "tick": 123,
    "entities": [
      {
        "id": "h_1",
        "type": "hunter", // 或 "prey"
        "x": 100.0, "y": 200.0,
        "angle": 1.57, // 朝向，弧度
        "speed": 32.0, // 线速度（像素/秒）
        "angular_velocity": 0.1, // 角速度（弧度/秒）
        "radius": 8.0,
        "energy": 85.0,
        "digestion": 0.0, // 消化计时，>0 表示消化中
        "age": 12.3,
        "generation": 2,
        "offspring_count": 1,
        "fov_deg": 40.0,
        "fov_range": 220.0,
        "rays": [ // 可选，如果后端已计算
          {"angle": 1.2, "distance": 80.0, "hit_type": "prey", "hit_id": "p_7"}
        ]
      }
    ]
  }
  ```

可视化与交互
- 点击实体显示调试面板：
  - 捕食者：剩余能量、迭代(tick)、子代个数、速度、分裂能量、视野范围、目光跟踪目标。
  - 非捕食者：能量、速度、分裂能量、生存时间、代数、子代范围。
- 快捷键：
  - `R` 切换显示 FOV 射线
  - `D` 切换调试面板
  - `P` 暂停/继续
  - `[`/`]` 调整 FOV 范围；`-`/`=` 调整射线数量

开发提示
- 若后端不提供 rays，前端会基于实体几何进行近似射线与碰撞可视化，仅用于调试。
- 为保证运动平滑，前端对位置与角度进行插值（lerp），并加入轻微的蠕动效果。
## 部署与运行（macOS/Linux）

### 环境准备
- 需要安装 `python3` 与 `git`。
- 推荐 Python 版本：3.11 及以上（已在 3.13 与 `pygame-ce 2.5.2` 验证）。

### 一键启动（推荐）

```bash
bash scripts/run.sh
```

脚本会自动：
- 创建虚拟环境 `.venv/`（若不存在）；
- 安装 `requirements.txt` 依赖；
- 使用虚拟环境 Python 运行 `src/app.py`。

### 手动步骤

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python src/app.py
```

### 操作与调试
- 点击实体以在左上角面板查看详细信息（含“存活时间”）。
- 快捷键：
  - `p`：暂停/继续
  - `r`：显示/隐藏射线
  - `d`：显示/隐藏调试面板
  - `[` / `]`：调整 FOV 距离缩放
  - `-` / `=`：减少/增加每实体射线数量

### 数据源切换
- 默认数据源为 `MockSource`（随机生成）：`src/app.py` 中已设置。
- 若需从 JSON 文件喂数据：在 `src/app.py` 中将

```python
# source = FileJSONSource(path="runtime/world.json")
source = MockSource(n_hunters=8, n_prey=24)
```

修改为：

```python
source = FileJSONSource(path="runtime/world.json")
# source = MockSource(n_hunters=8, n_prey=24)
```

并在 `runtime/world.json` 放置符合 `WorldState.from_dict` 规范的数据。

## 关键函数速览
- `src/app.py`
  - `main()`：主循环，拉取数据源帧、处理事件、绘制并刷新显示。
- `src/datasource.py`
  - `MockSource.poll()`：生成每帧世界状态（运动、分裂、射线更新）。
  - `_update_motion(dt)`：实体运动与能量/年龄更新、分裂触发逻辑。
  - `_spawn_child(parent, etype)`：创建子体，初始化半径成长与冷却。
  - `_compute_rays(e)`：前端近似 FOV 射线与碰撞估计。
- `src/render.py`
  - `PygameRenderer.handle_events(world)`：处理鼠标和键盘事件。
  - `PygameRenderer.draw_world(world)`：绘制网格、实体、调试面板。
  - `_draw_entity(e, gaze_dir, gaze_dist)`：平滑位置、拉伸形变与朝向指示。
  - `_pick_entity(world, pos)`：平滑位置下的点击命中检测。
  - `_fmt_age(age_seconds)`：将秒数格式化为 `mm:ss.ss`。
- `src/models.py`
  - `WorldState` / `EntityState` / `RayHit`：前后端数据结构定义与解析。

## 上传到 GitHub 指南

### 创建远程仓库
1. 登入 GitHub，新建一个空仓库（不勾选初始化 README/License）。
2. 复制仓库地址（HTTPS 或 SSH）。

### 本地首推
在项目根目录执行：

```bash
git init
git add .
git commit -m "Initial import"

# 远端地址二选一：
git remote add origin https://github.com/<your-user>/<your-repo>.git
# 或者（推荐已配置 SSH Key）：
git remote add origin git@github.com:<your-user>/<your-repo>.git

git branch -M main
git push -u origin main
```

### 后续更新

```bash
git add -A
git commit -m "Update"
git push
```

### 同学部署
同学在任意机器上：

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
bash scripts/run.sh
```

若使用 Windows，请在 PowerShell 中执行手动步骤的等价指令：

```powershell
python -m venv .venv
./.venv/Scripts/python -m pip install -r requirements.txt
./.venv/Scripts/python src/app.py
```