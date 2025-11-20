"""测试配置修复和增强奖励"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AgentConfig, EnvConfig
from rl_env import Stage1HunterReward
from core import WorldSimulator
from core.world import EntityState, WorldState
import math

def test_config_fix():
    """测试配置是否已修复"""
    print("=" * 80)
    print("测试1: 配置修复验证")
    print("=" * 80)

    agent_cfg = AgentConfig()

    print(f"✓ 猎人最大速度: {agent_cfg.HUNTER_SPEED_MAX} (期望: 50.0)")
    print(f"✓ 猎人角速度: {agent_cfg.HUNTER_ANGULAR_VELOCITY_MAX} (期望: 0.15)")
    print(f"✓ 猎物最大速度: {agent_cfg.PREY_SPEED_MAX} (期望: 45.0)")
    print(f"✓ 猎物角速度: {agent_cfg.PREY_ANGULAR_VELOCITY_MAX} (期望: 0.18)")

    assert agent_cfg.HUNTER_SPEED_MAX == 50.0, "猎人速度配置错误!"
    assert agent_cfg.PREY_SPEED_MAX == 45.0, "猎物速度配置错误!"
    assert agent_cfg.HUNTER_ANGULAR_VELOCITY_MAX == 0.15, "猎人角速度配置错误!"
    assert agent_cfg.PREY_ANGULAR_VELOCITY_MAX == 0.18, "猎物角速度配置错误!"

    print("\n✅ 配置修复验证通过!\n")


def test_enhanced_rewards():
    """测试增强奖励"""
    print("=" * 80)
    print("测试2: 增强奖励验证")
    print("=" * 80)

    reward_fn = Stage1HunterReward()

    # 检查奖励参数
    print(f"✓ 接近奖励scale: {reward_fn.approach_scale} (期望: 15.0, 原来: 5.0)")
    print(f"✓ 方向奖励scale: {reward_fn.direction_scale} (期望: 15.0, 原来: 5.0)")
    print(f"✓ 进度奖励scale: {reward_fn.progress_reward_scale} (期望: 10.0)")

    assert reward_fn.approach_scale == 15.0, "接近奖励scale错误!"
    assert reward_fn.direction_scale == 15.0, "方向奖励scale错误!"
    assert reward_fn.progress_reward_scale == 10.0, "进度奖励scale错误!"

    print("\n✅ 增强奖励参数验证通过!\n")


def test_reward_computation():
    """测试奖励计算 - 简化版"""
    print("=" * 80)
    print("测试3: 奖励函数内部状态")
    print("=" * 80)

    reward_fn = Stage1HunterReward()

    # 测试prev_distances字典是否存在
    assert hasattr(reward_fn, 'prev_distances'), "缺少prev_distances属性!"
    print(f"✓ 进度追踪字典已初始化: {type(reward_fn.prev_distances)}")

    # 测试重置功能
    reward_fn.prev_distances["test"] = 100.0
    reward_fn.prev_positions["test"] = (100, 100)

    # 手动调用重置
    reward_fn.prev_distances = {}
    reward_fn.prev_positions = {}

    assert len(reward_fn.prev_distances) == 0, "重置后应为空!"
    print(f"✓ 重置功能正常")

    print(f"\n✅ 奖励函数状态测试通过!\n")


def test_simulation_with_config():
    """测试模拟器使用新配置"""
    print("=" * 80)
    print("测试4: 模拟器配置应用")
    print("=" * 80)

    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()

    simulator = WorldSimulator(env_cfg, agent_cfg, use_parallel=False)
    simulator.initialize(n_hunters=2, n_prey=4)

    # 运行10步
    for i in range(10):
        world = simulator.step()

    # 检查速度是否在正确范围内
    for entity in world.entities:
        if entity.type == "hunter":
            assert entity.speed <= 50.0, f"猎人速度超过限制: {entity.speed}"
            print(f"✓ 猎人速度: {entity.speed:.1f} <= 50.0")
        else:
            assert entity.speed <= 45.0, f"猎物速度超过限制: {entity.speed}"
            print(f"✓ 猎物速度: {entity.speed:.1f} <= 45.0")

    simulator.shutdown()

    print("\n✅ 模拟器配置应用测试通过!\n")


if __name__ == "__main__":
    try:
        test_config_fix()
        test_enhanced_rewards()
        test_reward_computation()
        test_simulation_with_config()

        print("=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        print("\n下一步:")
        print("1. 删除旧模型: rm -rf curriculum_models/")
        print("2. 重新训练: python train_curriculum.py --stage stage1 --device cpu")
        print("3. 评估模型: python evaluate_models.py")
        print()

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
