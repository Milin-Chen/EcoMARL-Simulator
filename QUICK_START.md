# 快速开始指南

**5分钟快速上手 EcoMARL-Simulator**

---

## 📦 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd EcoMARL-Simulator
```

### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 最小依赖（仅运行可视化）

```bash
pip install numpy pygame
```

---

## 🎮 运行可视化演示

### 方式1: 使用训练好的模型（推荐）

```bash
python main.py
```

**效果**: 启动PyGame窗口，显示智能体使用训练好的模型进行追逐和逃跑

**控制**:
- 鼠标点击：选择智能体查看视野
- 空格键：暂停/继续
- ESC/关闭窗口：退出

### 方式2: 不使用模型（脚本行为）

```bash
python main.py --no-models
```

**效果**: 智能体使用简单的脚本行为（追击/逃跑）

### 方式3: 串行模式（调试用）

```bash
python main.py serial
```

**效果**: 禁用并行优化，方便调试

### 方式4: 无头模式（性能测试）

```bash
python main.py headless
```

**效果**: 不启动图形界面，仅输出性能统计

---

## 🤖 训练自己的模型

### 课程学习训练（推荐）

4阶段渐进式训练系统：

```bash
# Stage 1: 训练猎人对抗静止猎物
python train_curriculum.py --stage stage1

# Stage 2: 训练猎人对抗脚本猎物
python train_curriculum.py --stage stage2

# Stage 3: 训练猎物对抗智能猎人
python train_curriculum.py --stage stage3

# Stage 4: 联合微调（完整生态）
python train_curriculum.py --stage stage4
```

**模型保存位置**: `curriculum_models/`

### HPO增强训练（高级）

启用超参数优化增强功能：

```bash
python train_curriculum.py --stage stage1 --enable_hpo
```

**HPO特性**:
- 自适应奖励缩放
- 对抗平衡
- 距离进度追踪

**模型保存位置**: `curriculum_models_hpo/`

### 交互式快速训练

```bash
python train_simple.py
```

**效果**: 交互式菜单选择训练阶段和参数

---

## 📊 可视化训练结果

### 查看训练好的模型演示

```bash
python demo_curriculum_models.py
```

**效果**: 自动查找并演示最新训练的模型

---

## 🎯 常见使用场景

### 场景1: 我想看看效果

```bash
python main.py
```

### 场景2: 我想训练一个简单的猎人模型

```bash
python train_curriculum.py --stage stage1
```

训练完成后：

```bash
python main.py  # 自动加载最新模型
```

### 场景3: 我想完整训练一个生态系统

```bash
# 依次训练所有阶段
python train_curriculum.py --stage stage1
python train_curriculum.py --stage stage2
python train_curriculum.py --stage stage3
python train_curriculum.py --stage stage4

# 或者一次性训练所有阶段
python train_curriculum.py --stages stage1 stage2 stage3 stage4
```

### 场景4: 我想自定义参数

编辑配置文件：
- `config/env_config.py` - 环境参数（世界大小、能量系统）
- `config/agent_config.py` - 智能体参数（速度、视野）
- `config/training_config.py` - 训练参数（学习率、步数）

### 场景5: 我想测试性能

```bash
python main.py headless
```

查看FPS和帧时输出。

---

## 📁 项目结构速览

```
EcoMARL-Simulator/
├── main.py                    # 主可视化入口
├── train_curriculum.py        # 课程学习训练脚本
├── train_simple.py           # 交互式训练
├── demo_curriculum_models.py # 模型演示
│
├── config/                   # 配置模块
│   ├── env_config.py        # 环境配置
│   ├── agent_config.py      # 智能体配置
│   ├── render_config.py     # 渲染配置
│   └── training_config.py   # 训练配置
│
├── core/                     # 物理引擎
│   ├── world.py             # 世界模拟器
│   ├── physics.py           # 运动物理
│   ├── sensors.py           # 视野系统
│   └── energy.py            # 能量系统
│
├── rl_env/                   # 强化学习环境
│   ├── envs/                # Gym环境
│   ├── rewards/             # 奖励函数
│   ├── training/            # 训练组件
│   ├── observations.py      # 观测空间
│   └── agent_controller.py  # 智能体控制器
│
├── frontend/                 # 可视化
│   └── pygame_renderer.py   # PyGame渲染器
│
├── parallel/                 # 并行优化
│   ├── quadtree.py          # 空间索引
│   └── renderer.py          # 并行渲染
│
└── models/                   # 数据模型
    └── state.py             # 实体状态
```

---

## 🔧 常见问题

### Q1: 如何更改智能体数量？

**A**: 编辑 `config/env_config.py`:
```python
MAX_ENTITIES = 200  # 最大实体数量
```

或在代码中：
```python
simulator.initialize(n_hunters=10, n_prey=40)
```

### Q2: 训练太慢怎么办？

**A**:
1. 减少环境数量：`--n_envs 2`
2. 减少训练步数：编辑 `config/training_config.py`
3. 使用CPU: `--device cpu`（多核并行）

### Q3: 可视化界面卡顿？

**A**:
1. 减少实体数量
2. 使用串行模式：`python main.py serial`
3. 关闭调试信息：编辑 `config/render_config.py`

### Q4: 如何使用GPU训练？

**A**:
```bash
python train_curriculum.py --stage stage1 --device cuda
```

确保安装了CUDA版本的PyTorch。

### Q5: 模型保存在哪里？

**A**:
- 标准训练：`curriculum_models/`
- HPO训练：`curriculum_models_hpo/`
- 格式：`stageX_hunter_final.zip` 或 `stageX_prey_final.zip`

---

## 📚 下一步阅读

**新手推荐**:
1. [README.md](README.md) - 项目详细介绍
2. [docs/modules/CORE_MODULES.md](docs/modules/CORE_MODULES.md) - 核心模块详解

**训练相关**:
1. [docs/modules/TRAINING_SYSTEM.md](docs/modules/TRAINING_SYSTEM.md) - 训练系统详解
2. [docs/modules/RL_ENVIRONMENT.md](docs/modules/RL_ENVIRONMENT.md) - 强化学习环境

**进阶用户**:
1. [docs/modules/PARALLEL_OPTIMIZATION.md](docs/modules/PARALLEL_OPTIMIZATION.md) - 性能优化
2. [docs/modules/VISUALIZATION.md](docs/modules/VISUALIZATION.md) - 可视化系统

---

## 🎉 立即开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行可视化
python main.py

# 3. 开始训练（可选）
python train_curriculum.py --stage stage1
```

**祝你探索愉快！** 🚀
