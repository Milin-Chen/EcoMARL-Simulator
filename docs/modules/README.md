# 模块文档索引

**EcoMARL-Simulator 详细模块文档**

---

## 📚 文档列表

### 核心文档（已完成）

| 文档 | 描述 | 复杂度 |
|------|------|--------|
| [CORE_MODULES.md](CORE_MODULES.md) | 核心物理引擎：WorldSimulator, Physics, Sensors, Energy | ⭐⭐⭐ |
| [TRAINING_SYSTEM.md](TRAINING_SYSTEM.md) | 训练系统：课程学习、HPO优化、训练配置 | ⭐⭐⭐⭐ |
| [RL_ENVIRONMENT.md](RL_ENVIRONMENT.md) | 强化学习环境：Gym接口、奖励函数、观测空间 | ⭐⭐⭐⭐ |
| [VISUALIZATION.md](VISUALIZATION.md) | 可视化系统：PyGame渲染器、交互控制 | ⭐⭐ |
| [PARALLEL_OPTIMIZATION.md](PARALLEL_OPTIMIZATION.md) | 性能优化：QuadTree、多线程 | ⭐⭐⭐ |
| [CONFIGURATION.md](CONFIGURATION.md) | 配置系统：所有参数详解 | ⭐⭐ |

---

## 🎯 推荐阅读路径

### 新用户（了解系统）

```
1. README.md (项目概述)
   ↓
2. QUICK_START.md (快速上手)
   ↓
3. VISUALIZATION.md (了解可视化)
   ↓
4. CORE_MODULES.md (理解底层)
```

### 训练用户（训练模型）

```
1. TRAINING_SYSTEM.md (训练系统)
   ↓
2. CONFIGURATION.md (调整参数)
   ↓
3. RL_ENVIRONMENT.md (理解环境)
```

### 开发者（扩展功能）

```
1. CORE_MODULES.md (核心架构)
   ↓
2. RL_ENVIRONMENT.md (环境实现)
   ↓
3. PARALLEL_OPTIMIZATION.md (性能优化)
   ↓
4. 源代码注释
```

---

## 📖 文档概览

### CORE_MODULES.md - 核心物理引擎

**内容**:
- WorldSimulator - 世界模拟器
- PhysicsEngine - 运动物理、碰撞检测
- SensorSystem - 视野系统
- EnergySystem - 能量管理
- 数据模型 - EntityState, WorldState

**适用人群**: 开发者、研究者

**关键概念**:
- 更新循环（step函数）
- QuadTree空间索引
- 扇形视野模型
- 能量平衡设计

---

### TRAINING_SYSTEM.md - 训练系统

**内容**:
- 课程学习4阶段
- HPO超参数优化
- 训练配置详解
- 使用指南和故障排除

**适用人群**: 所有用户

**关键概念**:
- 渐进式训练
- 自适应奖励缩放
- 对抗平衡
- 持续追击/逃跑加成

---

### RL_ENVIRONMENT.md - 强化学习环境

**内容**:
- Gym环境接口
- 观测空间设计
- 奖励函数体系
- 动作空间

**适用人群**: 研究者、RL开发者

**关键概念**:
- MultiAgent环境
- 归一化观测
- 分层奖励设计
- 课程学习奖励

---

### VISUALIZATION.md - 可视化系统

**内容**:
- PyGame渲染器
- 交互控制
- 调试可视化
- 性能优化

**适用人群**: 所有用户

**关键概念**:
- 实时渲染
- 智能体视角切换
- 视野可视化
- FPS优化

---

### PARALLEL_OPTIMIZATION.md - 性能优化

**内容**:
- QuadTree空间索引
- 多线程并行
- 性能基准测试
- 优化策略

**适用人群**: 开发者

**关键概念**:
- O(n²) → O(n log n)
- 空间分割
- 并行碰撞检测
- 批量处理

---

### CONFIGURATION.md - 配置系统

**内容**:
- EnvConfig - 环境配置
- AgentConfig - 智能体配置
- RenderConfig - 渲染配置
- TrainingConfig - 训练配置

**适用人群**: 所有用户

**关键概念**:
- 统一配置架构
- 参数影响分析
- 常用配置模板
- 调优技巧

---

## 🔍 快速查找

### 如何...

| 问题 | 查看文档 | 章节 |
|------|---------|------|
| 运行可视化 | QUICK_START.md | 运行可视化演示 |
| 训练模型 | TRAINING_SYSTEM.md | 5. 使用指南 |
| 修改智能体速度 | CONFIGURATION.md | AgentConfig |
| 自定义奖励函数 | RL_ENVIRONMENT.md | 奖励函数 |
| 提升性能 | PARALLEL_OPTIMIZATION.md | QuadTree |
| 理解物理引擎 | CORE_MODULES.md | 2-5章 |

### 概念索引

| 概念 | 位置 |
|------|------|
| **QuadTree** | PARALLEL_OPTIMIZATION.md, CORE_MODULES.md (碰撞检测) |
| **课程学习** | TRAINING_SYSTEM.md (2. 课程学习系统) |
| **HPO** | TRAINING_SYSTEM.md (3. HPO超参数优化) |
| **视野系统** | CORE_MODULES.md (4. SensorSystem) |
| **能量系统** | CORE_MODULES.md (5. EnergySystem) |
| **观测空间** | RL_ENVIRONMENT.md (观测空间) |
| **奖励函数** | RL_ENVIRONMENT.md (奖励函数) |
| **PPO参数** | TRAINING_SYSTEM.md (4.1 PPO参数) |

---

## 📝 文档状态

| 文档 | 状态 | 最后更新 |
|------|------|---------|
| README.md | ✅ 完成 | 2025-01-20 |
| QUICK_START.md | ✅ 完成 | 2025-01-20 |
| CORE_MODULES.md | ✅ 完成 | 2025-01-20 |
| TRAINING_SYSTEM.md | ✅ 完成 | 2025-01-20 |
| RL_ENVIRONMENT.md | 📝 简化版 | 2025-01-20 |
| VISUALIZATION.md | 📝 简化版 | 2025-01-20 |
| PARALLEL_OPTIMIZATION.md | 📝 简化版 | 2025-01-20 |
| CONFIGURATION.md | 📝 简化版 | 2025-01-20 |

**注**: 简化版文档包含核心内容，可通过查看源代码获取更多细节。

---

## 🤝 贡献文档

欢迎改进文档！请遵循：

1. **Markdown格式**: 使用标准Markdown
2. **代码示例**: 提供可运行的代码
3. **清晰结构**: 使用标题、表格、列表
4. **实用性**: 注重实际应用，而非理论
5. **更新日期**: 修改后更新文档日期

---

## 📧 反馈

文档问题或建议请提交到：
- **Issues**: [GitHub Issues](https://github.com/yourusername/EcoMARL-Simulator/issues)
- **标签**: `documentation`

---

**文档是理解项目的钥匙，欢迎探索！** 📚
