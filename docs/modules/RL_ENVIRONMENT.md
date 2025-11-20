# 强化学习环境详解

**RL Environment - Gym接口、奖励函数、观测空间**

---

## 📋 目录

- [1. 环境概述](#1-环境概述)
- [2. 观测空间](#2-观测空间)
- [3. 动作空间](#3-动作空间)
- [4. 奖励函数](#4-奖励函数)
- [5. 使用示例](#5-使用示例)

---

## 1. 环境概述

### 1.1 环境层次

```
rl_env/
├── envs/
│   ├── gym_env_enhanced.py          # 基础环境
│   ├── gym_env_curriculum.py        # 课程学习环境
│   └── gym_env_curriculum_hpo.py    # HPO增强环境
│
├── rewards/
│   ├── rewards_enhanced.py          # 基础奖励V1
│   ├── rewards_enhanced_v2.py       # 增强奖励V2
│   ├── rewards_curriculum.py        # 课程学习奖励
│   └── rewards_curriculum_hpo.py    # HPO增强奖励
│
└── observations.py                  # 观测空间提取器
```

### 1.2 环境选择

| 环境 | 用途 | 推荐度 |
|------|------|--------|
| `CurriculumEcoMARLEnv` | 课程学习训练 | ⭐⭐⭐⭐⭐ |
| `CurriculumEcoMARLEnvHPO` | HPO增强训练 | ⭐⭐⭐⭐ |
| `EnhancedEcoMARLEnv` | 基础训练/测试 | ⭐⭐⭐ |

---

## 2. 观测空间

### 2.1 观测向量（16维）

```python
observation = [
    # 自身状态 (4维)
    normalized_energy,      # [0, 1] 能量百分比
    normalized_speed,       # [0, 1] 速度百分比
    normalized_angular,     # [-1, 1] 角速度归一化
    agent_type_encoding,    # 0.0=猎人, 1.0=猎物

    # 最近目标 (6维)
    relative_x,            # [-1, 1] 相对x坐标
    relative_y,            # [-1, 1] 相对y坐标
    distance,              # [0, 1] 距离归一化
    angle_diff,            # [-1, 1] 角度差异
    target_speed,          # [0, 1] 目标速度
    target_type,           # 0.0=猎人, 1.0=猎物

    # 视野内统计 (6维)
    num_hunters_visible,   # [0, 1] 可见猎人数量归一化
    num_preys_visible,     # [0, 1] 可见猎物数量归一化
    avg_hunter_distance,   # [0, 1] 平均猎人距离
    avg_prey_distance,     # [0, 1] 平均猎物距离
    nearest_hunter_dist,   # [0, 1] 最近猎人距离
    nearest_prey_dist,     # [0, 1] 最近猎物距离
]
```

### 2.2 观测提取

**实现**: `rl_env/observations.py`

```python
from rl_env import ObservationSpace

obs_space = ObservationSpace(agent_config)
obs = obs_space.get_observation(entity, all_entities)
# obs.shape = (16,)
```

---

## 3. 动作空间

### 3.1 连续动作（2维）

```python
action = [
    speed_delta,          # [-1, 1] 加速/减速
    angular_velocity_delta  # [-1, 1] 左转/右转
]
```

### 3.2 动作应用

```python
# 速度更新
entity.speed = clip(
    entity.speed + speed_delta * SPEED_INCREMENT,
    0.0,
    MAX_SPEED
)

# 角速度更新
entity.angular_velocity = clip(
    entity.angular_velocity + angular_delta * ANGULAR_INCREMENT,
    -MAX_ANGULAR,
    MAX_ANGULAR
)
```

---

## 4. 奖励函数

### 4.1 奖励体系

#### 猎人奖励（Stage1HunterReward）

```python
总奖励 = 存活奖励 + 追击奖励 + 捕获奖励 + 能量惩罚
```

**分项**:
- **存活奖励**: `+1.0/步` (保持存活)
- **追击奖励**: `距离减少 * 2.0` (接近猎物)
- **捕获奖励**: `+50.0` (成功捕获)
- **能量惩罚**: `-能量消耗 * 0.1` (鼓励节能)

**持续追击加成** (HPO版本):
```python
chase_multiplier = 1.0 + 2.0 * (chase_streak / 10)
# chase_streak=10 → 3.0x倍数
```

#### 猎物奖励（Stage3PreyReward）

```python
总奖励 = 存活奖励 + 逃跑奖励 + 被捕惩罚
```

**分项**:
- **存活奖励**: `+2.0/步` (保持存活)
- **逃跑奖励**: `距离增加 * 3.0` (远离猎人)
- **被捕惩罚**: `-100.0` (被捕获)

**持续逃跑加成** (HPO版本):
```python
escape_multiplier = 1.0 + 2.0 * (escape_streak / 10)
# escape_streak=10 → 3.0x倍数
```

### 4.2 课程学习奖励

| 阶段 | 奖励函数 | 特点 |
|------|---------|------|
| Stage 1 | Stage1HunterReward | 简单追击，静止目标 |
| Stage 2 | Stage2HunterReward | 预测拦截，移动目标 |
| Stage 3 | Stage3PreyReward | 逃避策略 |
| Stage 4 | Stage4JointReward | 平衡对抗 |

### 4.3 自定义奖励

```python
from rl_env import Stage1HunterReward

class MyReward(Stage1HunterReward):
    def compute_reward(self, hunter, world_state, prev_state):
        # 调用父类基础奖励
        base_reward = super().compute_reward(hunter, world_state, prev_state)

        # 自定义奖励：奖励高能量
        energy_bonus = 0.0
        if hunter.energy > 80:
            energy_bonus = 2.0

        # 惩罚低速度（鼓励激进）
        speed_penalty = 0.0
        if hunter.speed < 20:
            speed_penalty = -1.0

        return base_reward + energy_bonus + speed_penalty
```

---

## 5. 使用示例

### 5.1 基础训练

```python
from rl_env import CurriculumEcoMARLEnv
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

# 保存
model.save("my_hunter_model")
```

### 5.2 环境交互

```python
# 重置环境
obs = env.reset()
# obs: Dict[agent_id, observation_vector]

done = False
while not done:
    # 随机动作（或使用模型）
    actions = {
        agent_id: env.action_space.sample()
        for agent_id in obs.keys()
    }

    # 执行步骤
    obs, rewards, dones, info = env.step(actions)

    # 检查终止
    done = all(dones.values())

env.close()
```

### 5.3 使用训练好的模型

```python
from stable_baselines3 import PPO

# 加载模型
hunter_model = PPO.load("curriculum_models/stage2_hunter_final.zip")

# 推理
obs = env.reset()
for _ in range(1000):
    # 只控制猎人
    hunter_actions = {}
    for agent_id, agent_obs in obs.items():
        if "hunter" in agent_id:
            action, _ = hunter_model.predict(agent_obs, deterministic=True)
            hunter_actions[agent_id] = action

    obs, rewards, dones, info = env.step(hunter_actions)
```

---

## 📚 相关文档

- [训练系统](TRAINING_SYSTEM.md) - 如何使用环境训练
- [核心模块](CORE_MODULES.md) - 底层物理引擎
- [配置系统](CONFIGURATION.md) - 环境参数配置

---

**掌握RL环境，打造智能决策！** 🤖
