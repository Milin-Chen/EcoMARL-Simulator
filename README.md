# EcoMARL-Simulator

**基于多智能体强化学习的捕食者-猎物生态系统模拟器**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 项目简介

EcoMARL-Simulator 是一个高性能的生态系统模拟器，使用**多智能体强化学习(MARL)**技术模拟捕食者-猎物之间的复杂交互和演化行为。

### ✨ 核心特性

- 🎓 **课程学习训练**: 4阶段渐进式训练系统，从简单到复杂
- 🔧 **HPO超参数优化**: 自适应奖励缩放、对抗平衡、进度追踪
- 🎮 **实时可视化**: PyGame交互式渲染，支持智能体视角切换
- ⚡ **并行加速**: QuadTree空间索引 + 多线程优化，~15倍性能提升
- 📊 **完整生态**: 能量系统、视野限制、追逐/逃跑行为
- 🧪 **模块化设计**: 前后端分离，易于扩展和测试

### 🎬 演示效果

![Demo](assets/demo.gif) *(如果有的话)*

**主要行为**:
- 🔴 猎人（红色）：追击猎物，消耗能量，捕获后补充能量
- 🔵 猎物（蓝色）：逃避猎人，能量耗尽死亡
- 👁️ 视野系统：智能体只能感知视野范围内的目标
- ⚡ 能量机制：运动消耗能量，猎人捕获猎物补充能量

---

## ⚡ 快速开始

### 最快5分钟体验

```bash
# 1. 克隆项目
git clone <repository-url>
cd EcoMARL-Simulator

# 2. 安装依赖
pip install numpy pygame torch stable-baselines3

# 3. 运行可视化
python main.py
```

**完整安装和使用指南**: [QUICK_START.md](QUICK_START.md)

---

## 📚 文档导航

### 📖 快速入门

| 文档 | 描述 | 适用人群 |
|------|------|---------|
| [QUICK_START.md](QUICK_START.md) | 5分钟快速上手 | 所有用户 |
| [docs/modules/CORE_MODULES.md](docs/modules/CORE_MODULES.md) | 核心模块介绍 | 开发者 |
| [docs/modules/TRAINING_SYSTEM.md](docs/modules/TRAINING_SYSTEM.md) | 训练系统详解 | 研究者 |

### 🔬 功能模块

| 模块 | 文档 | 功能 |
|------|------|------|
| **物理引擎** | [docs/modules/CORE_MODULES.md](docs/modules/CORE_MODULES.md) | 世界模拟、运动物理、碰撞检测 |
| **强化学习** | [docs/modules/RL_ENVIRONMENT.md](docs/modules/RL_ENVIRONMENT.md) | Gym环境、奖励函数、观测空间 |
| **课程学习** | [docs/modules/TRAINING_SYSTEM.md](docs/modules/TRAINING_SYSTEM.md) | 4阶段训练、HPO优化 |
| **可视化** | [docs/modules/VISUALIZATION.md](docs/modules/VISUALIZATION.md) | PyGame渲染、交互控制 |
| **性能优化** | [docs/modules/PARALLEL_OPTIMIZATION.md](docs/modules/PARALLEL_OPTIMIZATION.md) | QuadTree、多线程 |
| **配置系统** | [docs/modules/CONFIGURATION.md](docs/modules/CONFIGURATION.md) | 环境、智能体、训练参数 |

---

## 🏗️ 系统架构

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户交互层                               │
│  main.py  train_curriculum.py  demo_curriculum_models.py   │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌────────▼──────────┐
│  可视化前端     │      │   强化学习环境     │
│  PyGame渲染    │      │  Gym接口/PPO训练  │
│  frontend/     │      │  rl_env/          │
└───────┬────────┘      └────────┬──────────┘
        │                        │
        └────────────┬───────────┘
                     │
           ┌─────────▼──────────┐
           │   核心物理引擎      │
           │  WorldSimulator    │
           │  Physics/Sensors   │
           │  core/             │
           └─────────┬──────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐      ┌────────▼──────────┐
│  并行优化       │      │   数据模型        │
│  QuadTree      │      │  EntityState      │
│  parallel/     │      │  models/          │
└────────────────┘      └───────────────────┘
```

### 核心组件

| 组件 | 路径 | 职责 |
|------|------|------|
| **配置系统** | `config/` | 环境、智能体、训练、渲染参数 |
| **物理引擎** | `core/` | 世界模拟、运动、传感器、能量 |
| **数据模型** | `models/` | 实体状态、世界状态 |
| **RL环境** | `rl_env/` | Gym环境、奖励、观测、训练 |
| **可视化** | `frontend/` | PyGame渲染器 |
| **并行优化** | `parallel/` | QuadTree、多线程 |

---

## 🎓 课程学习系统

### 4阶段训练流程

```
Stage 1: 猎人 vs 静止猎物
    ↓ 学会基础追击
Stage 2: 猎人 vs 脚本猎物
    ↓ 学会预测和拦截
Stage 3: 智能猎物训练
    ↓ 学会逃避策略
Stage 4: 联合微调
    ↓ 完整生态系统
```

### 训练命令

```bash
# 方式1: 单阶段训练
python train_curriculum.py --stage stage1

# 方式2: 连续训练多个阶段
python train_curriculum.py --stages stage1 stage2 stage3 stage4

# 方式3: 启用HPO增强
python train_curriculum.py --stage stage1 --enable_hpo

# 方式4: 交互式训练
python train_simple.py
```

**详细训练指南**: [docs/modules/TRAINING_SYSTEM.md](docs/modules/TRAINING_SYSTEM.md)

---

## 🔧 配置系统

### 统一配置架构

所有参数集中在 `config/` 目录：

```python
from config import EnvConfig, AgentConfig, TrainingConfig

# 环境配置
env_cfg = EnvConfig()
env_cfg.WINDOW_WIDTH = 1600
env_cfg.WINDOW_HEIGHT = 1000

# 智能体配置
agent_cfg = AgentConfig()
agent_cfg.HUNTER_SPEED_MAX = 50.0
agent_cfg.PREY_SPEED_MAX = 45.0

# 训练配置
train_cfg = TrainingConfig()
stage1 = train_cfg.get_stage_config("stage1")
print(stage1.total_timesteps)  # 50000
```

### 配置文件

| 文件 | 内容 | 示例参数 |
|------|------|---------|
| `env_config.py` | 环境参数 | 世界大小、最大实体数、能量系统 |
| `agent_config.py` | 智能体参数 | 速度、视野、转向速度 |
| `render_config.py` | 渲染参数 | FPS、颜色、调试选项 |
| `training_config.py` | 训练参数 | 学习率、批次大小、训练步数 |

**详细配置说明**: [docs/modules/CONFIGURATION.md](docs/modules/CONFIGURATION.md)

---

## ⚡ 性能优化

### QuadTree空间索引

**性能对比**:

| 实体数 | 线性查找 | QuadTree | 提升 |
|-------|---------|----------|------|
| 20    | 400次   | ~86次    | 5x   |
| 100   | 10,000次| ~664次   | 15x  |
| 200   | 40,000次| ~1,529次 | 26x  |

**启用方式**:
```bash
# 默认启用
python main.py

# 禁用（调试用）
python main.py serial
```

### 多线程并行

- **后端**: `ParallelRenderer` 用于碰撞检测和传感器查询
- **前端**: `PygameRenderer` 使用QuadTree加速最近邻查找

**详细优化说明**: [docs/modules/PARALLEL_OPTIMIZATION.md](docs/modules/PARALLEL_OPTIMIZATION.md)

---

## 📊 使用示例

### 示例1: 基础可视化

```python
from core import WorldSimulator
from frontend import PygameRenderer
from config import EnvConfig, AgentConfig

# 创建模拟器
env_cfg = EnvConfig()
agent_cfg = AgentConfig()
simulator = WorldSimulator(env_cfg, agent_cfg, use_parallel=True)
simulator.initialize(n_hunters=6, n_prey=18)

# 创建渲染器并运行
renderer = PygameRenderer(use_parallel=True)

class DataSource:
    def poll(self):
        return simulator.step()
    def get_performance_stats(self):
        return simulator.get_stats()
    def shutdown(self):
        simulator.shutdown()

source = DataSource()
renderer.run_loop(source)
```

### 示例2: RL训练

```python
from rl_env import CurriculumEcoMARLEnv
from config import TrainingConfig
from stable_baselines3 import PPO

# 创建环境
env = CurriculumEcoMARLEnv(
    stage="stage1",
    n_hunters=3,
    n_prey=6,
)

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=50000)

# 保存模型
model.save("my_model")
```

### 示例3: 自定义奖励函数

```python
from rl_env.rewards import Stage1HunterReward

class MyCustomReward(Stage1HunterReward):
    def compute_reward(self, hunter, world_state, prev_state):
        # 基础奖励
        reward = super().compute_reward(hunter, world_state, prev_state)

        # 自定义奖励：奖励保持高能量
        if hunter.energy > 80:
            reward += 1.0

        return reward
```

---

## 🧪 测试

```bash
# 配置统一性测试
python tests/test_config_unified.py

# 训练系统测试
python tests/test_unified_training.py

# 性能基准测试
python tests/benchmark_performance.py
```

---

## 📁 项目结构

```
EcoMARL-Simulator/
├── README.md                      # 项目主文档（本文件）
├── QUICK_START.md                 # 快速开始指南
├── requirements.txt               # Python依赖
│
├── main.py                        # 主可视化入口
├── train_curriculum.py            # 课程学习训练（统一版）
├── train_simple.py               # 交互式训练
├── train.py                      # 基础PPO训练
├── demo_curriculum_models.py     # 模型演示脚本
│
├── config/                        # 配置模块
│   ├── env_config.py             # 环境配置
│   ├── agent_config.py           # 智能体配置
│   ├── render_config.py          # 渲染配置
│   └── training_config.py        # 训练配置 ✨统一
│
├── core/                          # 核心物理引擎
│   ├── world.py                  # WorldSimulator - 世界模拟
│   ├── physics.py                # PhysicsEngine - 运动物理
│   ├── sensors.py                # SensorSystem - 视野系统
│   └── energy.py                 # EnergySystem - 能量管理
│
├── models/                        # 数据模型
│   └── state.py                  # EntityState, WorldState
│
├── rl_env/                        # 强化学习环境
│   ├── envs/                     # Gym环境实现
│   │   ├── gym_env_enhanced.py          # 基础环境
│   │   ├── gym_env_curriculum.py        # 课程学习环境
│   │   └── gym_env_curriculum_hpo.py    # HPO增强环境
│   │
│   ├── rewards/                  # 奖励函数
│   │   ├── rewards_enhanced.py          # 基础奖励V1
│   │   ├── rewards_enhanced_v2.py       # 增强奖励V2
│   │   ├── rewards_curriculum.py        # 课程学习奖励
│   │   ├── rewards_curriculum_hpo.py    # HPO增强奖励
│   │   └── hpo_enhancements.py          # HPO组件
│   │
│   ├── training/                 # 训练组件
│   │   ├── networks.py           # ActorCriticNetwork
│   │   └── ppo_trainer.py        # PPOTrainer
│   │
│   ├── observations.py           # 观测空间
│   └── agent_controller.py       # 智能体控制器
│
├── frontend/                      # 可视化前端
│   └── pygame_renderer.py        # PyGame渲染器
│
├── parallel/                      # 并行优化
│   ├── quadtree.py               # QuadTree空间索引
│   ├── scheduler.py              # 任务调度
│   └── renderer.py               # 并行渲染
│
├── tests/                         # 测试套件
│   ├── test_config_unified.py    # 配置统一性测试
│   ├── test_unified_training.py  # 训练系统测试
│   └── benchmark_performance.py  # 性能基准测试
│
├── docs/                          # 文档目录
│   └── modules/                  # 模块详细文档
│       ├── CORE_MODULES.md       # 核心模块
│       ├── RL_ENVIRONMENT.md     # 强化学习环境
│       ├── TRAINING_SYSTEM.md    # 训练系统
│       ├── VISUALIZATION.md      # 可视化系统
│       ├── PARALLEL_OPTIMIZATION.md  # 性能优化
│       └── CONFIGURATION.md      # 配置系统
│
├── curriculum_models/             # 标准训练模型
└── curriculum_models_hpo/         # HPO训练模型
```

---

## 🔬 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| **语言** | Python 3.8+ | 主要开发语言 |
| **深度学习** | PyTorch 2.0+ | 神经网络 |
| **强化学习** | Stable-Baselines3 | PPO算法 |
| **可视化** | PyGame 2.0+ | 实时渲染 |
| **数值计算** | NumPy | 数组运算 |
| **并行** | multiprocessing | 多线程 |
| **空间索引** | QuadTree (自实现) | 性能优化 |

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

**开发规范**:
- 遵循PEP 8代码风格
- 添加单元测试
- 更新相关文档

---

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- **Stable-Baselines3**: PPO算法实现
- **PyGame**: 可视化框架
- **PyTorch**: 深度学习框架

---

## 📧 联系方式

- **项目链接**: [GitHub Repository](https://github.com/yourusername/EcoMARL-Simulator)
- **问题反馈**: [Issues](https://github.com/yourusername/EcoMARL-Simulator/issues)

---

## 🚀 立即开始

```bash
# 克隆项目
git clone <repository-url>
cd EcoMARL-Simulator

# 安装依赖
pip install -r requirements.txt

# 运行可视化
python main.py

# 开始训练
python train_curriculum.py --stage stage1
```

**更多详情**: [QUICK_START.md](QUICK_START.md)

---

**祝你探索愉快！** 🎉
