"""测试演示程序中的模型动作"""

from demo_curriculum_models import CurriculumModelController, DataSourceWithModel
from core import WorldSimulator
from config import EnvConfig, AgentConfig
import os

# 配置
env_cfg = EnvConfig()
agent_cfg = AgentConfig()

# 使用优化参数
agent_cfg.HUNTER_SPEED_MAX = 50.0
agent_cfg.PREY_SPEED_MAX = 45.0

# 加载模型
hunter_model = "curriculum_models/stage1_hunter_final.zip"
prey_model = None  # 使用脚本

model_controller = CurriculumModelController(hunter_model, prey_model)

# 初始化模拟器
simulator = WorldSimulator(env_cfg, agent_cfg, use_parallel=False)
simulator.initialize(n_hunters=3, n_prey=6)

print("=" * 80)
print("测试模型动作应用")
print("=" * 80)

# 创建数据源
source = DataSourceWithModel(simulator, model_controller)

# 运行100步并记录
predation_count = 0
for step in range(100):
    world = source.poll()

    # 检查捕食
    for event in world.events:
        if event.type == "predation":
            predation_count += 1
            print(f"\n步{step}: 捕食事件! hunter={event.hunter_id[:8]}, prey={event.prey_id[:8]}")

    # 每20步打印状态
    if step % 20 == 0:
        hunters = [e for e in world.entities if e.type == "hunter"]
        preys = [e for e in world.entities if e.type == "prey"]

        print(f"\n步{step}:")
        print(f"  猎人数量: {len(hunters)}, 猎物数量: {len(preys)}")

        if hunters:
            h = hunters[0]
            print(f"  猎人示例: speed={h.speed:.2f}, angular_v={h.angular_velocity:.3f}, pos=({h.x:.1f}, {h.y:.1f})")

        if preys:
            p = preys[0]
            print(f"  猎物示例: speed={p.speed:.2f}, angular_v={p.angular_velocity:.3f}, pos=({p.x:.1f}, {p.y:.1f})")

print("\n" + "=" * 80)
print(f"测试完成! 共发生 {predation_count} 次捕食")
print("=" * 80)

source.shutdown()
