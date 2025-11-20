# 配置系统详解

**Configuration System - 统一参数管理**

---

## 📋 配置文件

所有配置集中在 `config/` 目录：

| 文件 | 内容 | 主要参数 |
|------|------|---------|
| `env_config.py` | 环境配置 | 世界大小、能量系统、捕获半径 |
| `agent_config.py` | 智能体配置 | 速度、视野、转向速度 |
| `render_config.py` | 渲染配置 | FPS、颜色、调试选项 |
| `training_config.py` | 训练配置 | 学习率、PPO参数、课程阶段 |

---

## 1. EnvConfig - 环境配置

**文件**: `config/env_config.py`

```python
from config import EnvConfig

env_cfg = EnvConfig()

# 世界尺寸
env_cfg.WINDOW_WIDTH = 1600   # 默认1600
env_cfg.WINDOW_HEIGHT = 1000  # 默认1000

# 实体限制
env_cfg.MAX_ENTITIES = 200    # 最大实体数

# 能量系统
env_cfg.ENERGY_DECAY_RATE = 0.05        # 基础代谢
env_cfg.SPEED_ENERGY_COST = 0.01        # 速度消耗
env_cfg.TURN_ENERGY_COST = 0.02         # 转向消耗
env_cfg.CAPTURE_ENERGY_GAIN = 30.0      # 捕获补充

# 物理参数
env_cfg.CAPTURE_RADIUS = 15.0           # 捕获半径
env_cfg.DEFAULT_RADIUS = 8.0            # 实体半径
env_cfg.FRICTION_COEFFICIENT = 0.98     # 摩擦系数
```

---

## 2. AgentConfig - 智能体配置

**文件**: `config/agent_config.py`

```python
from config import AgentConfig

agent_cfg = AgentConfig()

# 猎人参数
agent_cfg.HUNTER_SPEED_MAX = 50.0               # 最大速度
agent_cfg.HUNTER_ANGULAR_VELOCITY_MAX = 0.15    # 最大角速度
agent_cfg.HUNTER_FOV_DEG = 120.0                # 视野角度
agent_cfg.HUNTER_VIEW_DISTANCE = 250.0          # 视距

# 猎物参数
agent_cfg.PREY_SPEED_MAX = 45.0                 # 最大速度
agent_cfg.PREY_ANGULAR_VELOCITY_MAX = 0.18      # 最大角速度
agent_cfg.PREY_FOV_DEG = 150.0                  # 视野角度
agent_cfg.PREY_VIEW_DISTANCE = 300.0            # 视距
```

---

## 3. RenderConfig - 渲染配置

**文件**: `config/render_config.py`

```python
from config import RenderConfig

render_cfg = RenderConfig()

# 性能
render_cfg.FPS = 60                  # 帧率

# 颜色
render_cfg.HUNTER_COLOR = (200, 50, 50)   # 猎人颜色（红）
render_cfg.PREY_COLOR = (50, 100, 200)    # 猎物颜色（蓝）
render_cfg.BG_COLOR = (30, 30, 40)        # 背景颜色

# 调试
render_cfg.SHOW_FOV = True           # 显示视野
render_cfg.SHOW_DEBUG_INFO = True    # 显示调试信息
```

---

## 4. TrainingConfig - 训练配置

**文件**: `config/training_config.py`

```python
from config import TrainingConfig

train_cfg = TrainingConfig()

# PPO参数
train_cfg.PPO_N_STEPS = 2048         # 每次更新步数
train_cfg.PPO_BATCH_SIZE = 64        # 批次大小
train_cfg.PPO_N_EPOCHS = 10          # Epoch数
train_cfg.PPO_LEARNING_RATE = 3e-4   # 学习率
train_cfg.PPO_GAMMA = 0.99           # 折扣因子

# 获取阶段配置
stage1 = train_cfg.get_stage_config("stage1")
print(stage1.total_timesteps)  # 50000
print(stage1.learning_rate)    # 3e-4
```

---

## 📚 使用示例

```python
from config import EnvConfig, AgentConfig, TrainingConfig
from core import WorldSimulator

# 创建配置
env_cfg = EnvConfig()
agent_cfg = AgentConfig()

# 自定义参数
env_cfg.WINDOW_WIDTH = 2000
agent_cfg.HUNTER_SPEED_MAX = 60.0

# 应用到模拟器
simulator = WorldSimulator(
    env_config=env_cfg,
    agent_config=agent_cfg
)
```

---

**统一配置，高效管理！** ⚙️
