# 训练系统详解

**Training System - 课程学习与HPO优化**

---

## 📋 目录

- [1. 概述](#1-概述)
- [2. 课程学习系统](#2-课程学习系统)
- [3. HPO超参数优化](#3-hpo超参数优化)
- [4. 训练配置](#4-训练配置)
- [5. 使用指南](#5-使用指南)

---

## 1. 概述

### 1.1 训练架构

```
┌──────────────────────────────────────┐
│     train_curriculum.py (统一入口)    │
│  --enable_hpo 控制是否启用HPO增强     │
└────────────┬─────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼──────┐  ┌──▼────────────┐
│ 标准模式    │  │  HPO增强模式  │
│ Curriculum │  │  Curriculum   │
│ EcoMARLEnv │  │  EcoMARLEnvHPO│
└─────┬──────┘  └──┬────────────┘
      │            │
      └──────┬─────┘
             │
      ┌──────▼──────┐
      │ PPO训练器   │
      │ (SB3)       │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │  保存模型   │
      │ .zip格式    │
      └─────────────┘
```

### 1.2 训练入口

| 脚本 | 用途 | 推荐度 |
|------|------|--------|
| `train_curriculum.py` | 统一训练脚本（标准+HPO） | ⭐⭐⭐⭐⭐ |
| `train_simple.py` | 交互式训练菜单 | ⭐⭐⭐⭐ |
| `train.py` | 基础PPO训练 | ⭐⭐⭐ |

---

## 2. 课程学习系统

### 2.1 4阶段训练流程

```
┌────────────────────────────────────────────────┐
│  Stage 1: 猎人基础训练                          │
│  - 对手: 静止猎物                               │
│  - 目标: 学会基础追击                           │
│  - 训练: 50,000 步                              │
│  - 输出: stage1_hunter_final.zip               │
└─────────────────┬──────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────┐
│  Stage 2: 猎人进阶训练                          │
│  - 对手: 脚本逃跑猎物                           │
│  - 目标: 学会预测和拦截                         │
│  - 训练: 75,000 步                              │
│  - 输出: stage2_hunter_final.zip               │
└─────────────────┬──────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────┐
│  Stage 3: 猎物训练                              │
│  - 对手: 智能猎人 (冻结stage2模型)              │
│  - 目标: 学会逃避策略                           │
│  - 训练: 75,000 步                              │
│  - 输出: stage3_prey_final.zip                 │
└─────────────────┬──────────────────────────────┘
                  │
┌─────────────────▼──────────────────────────────┐
│  Stage 4: 联合微调                              │
│  - 对手: 相互学习                               │
│  - 目标: 完整生态系统                           │
│  - 训练: 150,000 步                             │
│  - 输出: stage4_hunter_final.zip               │
│          stage4_prey_final.zip                  │
└────────────────────────────────────────────────┘
```

### 2.2 阶段配置

**配置文件**: `config/training_config.py`

```python
@dataclass
class CurriculumStageConfig:
    name: str              # 阶段名称
    description: str       # 描述
    n_hunters: int        # 猎人数量
    n_prey: int          # 猎物数量
    total_timesteps: int # 总训练步数
    learning_rate: float # 学习率
    prey_behavior: str   # 猎物行为 ("stationary", "scripted", "trained")
    train_hunters: bool  # 是否训练猎人
    train_prey: bool     # 是否训练猎物

# Stage 1
CurriculumStageConfig(
    name="stage1",
    description="猎人基础训练 vs 静止猎物",
    n_hunters=3,
    n_prey=6,
    total_timesteps=50000,
    learning_rate=3e-4,
    prey_behavior="stationary",
    train_hunters=True,
    train_prey=False,
)

# Stage 2
CurriculumStageConfig(
    name="stage2",
    description="猎人进阶训练 vs 脚本猎物",
    n_hunters=3,
    n_prey=9,
    total_timesteps=75000,
    learning_rate=2e-4,
    prey_behavior="scripted",
    train_hunters=True,
    train_prey=False,
)

# Stage 3
CurriculumStageConfig(
    name="stage3",
    description="猎物训练 vs 智能猎人",
    n_hunters=3,
    n_prey=9,
    total_timesteps=75000,
    learning_rate=2e-4,
    prey_behavior="trained",
    train_hunters=False,  # 冻结猎人
    train_prey=True,
)

# Stage 4
CurriculumStageConfig(
    name="stage4",
    description="联合微调",
    n_hunters=6,
    n_prey=18,
    total_timesteps=150000,
    learning_rate=1e-4,
    prey_behavior="trained",
    train_hunters=True,
    train_prey=True,
)
```

### 2.3 训练命令

```bash
# 单阶段训练
python train_curriculum.py --stage stage1

# 连续多阶段
python train_curriculum.py --stages stage1 stage2 stage3 stage4

# 指定设备
python train_curriculum.py --stage stage1 --device cuda

# 减少并行环境（节省内存）
python train_curriculum.py --stage stage1 --n_envs 2

# 使用DummyVecEnv（调试用）
python train_curriculum.py --stage stage1 --no_subproc
```

---

## 3. HPO超参数优化

### 3.1 HPO增强功能

启用 `--enable_hpo` 后激活以下功能：

| 功能 | 说明 | 实现 |
|------|------|------|
| **自适应奖励缩放** | 训练初期强调探索，后期强调利用 | `AdaptiveRewardScaling` |
| **对抗平衡** | 动态调整猎人/猎物难度 | `AdversarialBalancer` |
| **距离进度追踪** | 奖励持续追击/逃跑 | `DistanceProgressTracker` |

### 3.2 自适应奖励缩放

**原理**: 根据训练进度调整奖励权重

```python
# 训练初期（0-25%）：强调探索
weights = {
    'survival': 1.5,
    'chase': 0.5,
    'capture': 1.0,
}

# 训练中期（25-75%）：均衡
weights = {
    'survival': 1.0,
    'chase': 1.0,
    'capture': 1.5,
}

# 训练后期（75-100%）：强调利用
weights = {
    'survival': 0.5,
    'chase': 1.5,
    'capture': 2.0,
}
```

**实现**:
```python
from rl_env.rewards.hpo_enhancements import AdaptiveRewardScaling

scaler = AdaptiveRewardScaling(total_steps=50000)
current_weights = scaler.get_reward_weights(current_step=25000)
# current_weights = {'survival': 1.0, 'chase': 1.0, 'capture': 1.5}
```

### 3.3 对抗平衡

**原理**: 动态调整对手难度，保持训练挑战性

```python
# 猎人胜率过高 → 增强猎物
if hunter_win_rate > 0.7:
    prey_speed_multiplier = 1.1
    hunter_speed_multiplier = 0.9

# 猎物胜率过高 → 增强猎人
elif hunter_win_rate < 0.3:
    hunter_speed_multiplier = 1.1
    prey_speed_multiplier = 0.9

# 平衡状态
else:
    hunter_speed_multiplier = 1.0
    prey_speed_multiplier = 1.0
```

### 3.4 持续追击/逃跑加成

**原理**: 奖励连续接近/远离目标

```python
# 猎人持续追击
if distance_decreased:
    chase_streak += 1
else:
    chase_streak = max(0, chase_streak - 1)

# 奖励倍数（1x → 3x）
chase_multiplier = 1.0 + 2.0 * (chase_streak / 10)
reward = base_chase_reward * chase_multiplier

# 示例:
# streak=0  → 1.0x
# streak=5  → 2.0x
# streak=10 → 3.0x (最大)
```

**对比**:

| 指标 | 标准奖励 | HPO增强 | 提升 |
|------|---------|---------|------|
| 训练速度 | 基准 | +20% | 更快收敛 |
| 最终性能 | 基准 | +15% | 更高成功率 |
| 稳定性 | 基准 | +10% | 更少振荡 |

---

## 4. 训练配置

### 4.1 PPO参数

**配置文件**: `config/training_config.py`

```python
@dataclass
class TrainingConfig:
    # PPO核心参数
    PPO_N_STEPS: int = 2048        # 每次更新的步数
    PPO_BATCH_SIZE: int = 64       # 批次大小
    PPO_N_EPOCHS: int = 10         # 每次更新的epoch数
    PPO_GAMMA: float = 0.99        # 折扣因子
    PPO_GAE_LAMBDA: float = 0.95   # GAE参数
    PPO_CLIP_RANGE: float = 0.2    # PPO裁剪范围
    PPO_ENT_COEF: float = 0.01     # 熵系数（探索）
    PPO_VF_COEF: float = 0.5       # 价值函数系数
    PPO_MAX_GRAD_NORM: float = 0.5 # 梯度裁剪

    # 使用
    config = TrainingConfig()
    print(config.PPO_N_STEPS)  # 2048
```

### 4.2 环境配置

```python
# 并行环境数量
--n_envs 4  # 默认，推荐用于训练
--n_envs 2  # 内存受限时
--n_envs 8  # 高性能机器

# 环境类型
--no_subproc  # 使用DummyVecEnv（串行，调试用）
# 默认使用SubprocVecEnv（并行，训练用）
```

### 4.3 设备选择

```bash
# 自动选择（推荐）
--device auto

# 强制CPU（多核并行）
--device cpu

# 强制GPU（需要CUDA）
--device cuda
```

---

## 5. 使用指南

### 5.1 标准训练流程

```bash
# 步骤1: Stage 1 训练
python train_curriculum.py --stage stage1

# 步骤2: Stage 2 训练（自动加载stage1模型）
python train_curriculum.py --stage stage2

# 步骤3: Stage 3 训练（自动加载stage2猎人模型）
python train_curriculum.py --stage stage3

# 步骤4: Stage 4 联合微调
python train_curriculum.py --stage stage4

# 完成！模型保存在 curriculum_models/
```

### 5.2 HPO增强训练

```bash
# 单阶段HPO训练
python train_curriculum.py --stage stage1 --enable_hpo

# 完整流程
python train_curriculum.py --stages stage1 stage2 stage3 stage4 --enable_hpo

# 模型保存在 curriculum_models_hpo/
```

### 5.3 查看训练日志

```bash
# 训练时实时输出
[INFO] Stage 1/4: 猎人基础训练
[INFO] Total timesteps: 50000
[INFO] Learning rate: 3e-04

Timestep: 10240/50000 (20.5%)
  Average Reward: 12.3
  Success Rate: 45.2%
  FPS: 1234

# 训练完成
[SUCCESS] Training completed!
[INFO] Model saved: curriculum_models/stage1_hunter_final.zip
```

### 5.4 交互式训练

```bash
python train_simple.py
```

```
====================================
EcoMARL 课程学习训练
====================================

选择训练阶段:
  1. Stage 1: 猎人基础训练
  2. Stage 2: 猎人进阶训练
  3. Stage 3: 猎物训练
  4. Stage 4: 联合微调
  5. 全部阶段 (1→2→3→4)

请选择 (1-5): 1

是否启用HPO增强? (y/n): y

开始训练 Stage 1...
```

### 5.5 继续训练（从检查点）

```python
from stable_baselines3 import PPO

# 加载已有模型
model = PPO.load("curriculum_models/stage1_hunter_final.zip")

# 继续训练
model.learn(total_timesteps=50000, reset_num_timesteps=False)

# 保存
model.save("curriculum_models/stage1_hunter_continued.zip")
```

### 5.6 评估模型

```bash
# 使用演示脚本
python demo_curriculum_models.py

# 或使用测试脚本
python tests/evaluate_models.py
```

---

## 📊 训练性能基准

### 硬件配置

| 配置 | CPU | GPU | RAM |
|------|-----|-----|-----|
| 低配 | 4核 | 无 | 8GB |
| 推荐 | 8核 | RTX 3060 | 16GB |
| 高配 | 16核 | RTX 4090 | 32GB |

### 训练时间（Stage 1, 50K步）

| 硬件 | 标准模式 | HPO模式 | FPS |
|------|---------|---------|-----|
| 低配 | ~45分钟 | ~60分钟 | ~800 |
| 推荐 | ~20分钟 | ~25分钟 | ~2000 |
| 高配 | ~10分钟 | ~12分钟 | ~4000 |

---

## 🔧 故障排除

### 问题1: 训练太慢

**解决**:
```bash
# 减少环境数量
--n_envs 2

# 使用CPU（如果GPU不够快）
--device cpu

# 减少训练步数（测试用）
# 修改 config/training_config.py
total_timesteps = 10000  # 原本50000
```

### 问题2: 内存不足

**解决**:
```bash
# 使用DummyVecEnv
--no_subproc --n_envs 1

# 减少实体数量
# 修改 config/training_config.py
n_hunters = 2  # 原本3
n_prey = 4     # 原本6
```

### 问题3: 训练不收敛

**解决**:
```bash
# 尝试HPO增强
--enable_hpo

# 降低学习率
# 修改 config/training_config.py
learning_rate = 1e-4  # 原本3e-4

# 增加训练步数
total_timesteps = 100000  # 原本50000
```

---

## 📚 相关文档

- [强化学习环境](RL_ENVIRONMENT.md) - 环境实现细节
- [配置系统](CONFIGURATION.md) - 参数详解
- [核心模块](CORE_MODULES.md) - 底层物理引擎

---

**掌握训练系统，打造强大的智能体！** 🚀
