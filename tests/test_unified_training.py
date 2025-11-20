"""
测试统一训练脚本
Test Unified Training Script

验证:
1. TrainingConfig正确加载
2. HPO奖励函数包含持续追击/逃跑加成
3. 标准模式和HPO模式都能正常创建环境
"""

import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TrainingConfig, AgentConfig, EnvConfig
from rl_env import (
    EnhancedEcoMARLEnv,
    CurriculumEcoMARLEnv,
    CurriculumEcoMARLEnvHPO,
    Stage1HunterRewardHPO,
    Stage3PreyRewardHPO,
)


def test_training_config():
    """测试TrainingConfig加载"""
    print("\n=== 测试1: TrainingConfig加载 ===")

    config = TrainingConfig()
    print(f"✓ PPO参数:")
    print(f"  - n_steps: {config.PPO_N_STEPS}")
    print(f"  - batch_size: {config.PPO_BATCH_SIZE}")
    print(f"  - learning_rate默认: 见各阶段配置")

    stages = config.get_stage_configs()
    print(f"\n✓ 课程学习阶段: {list(stages.keys())}")

    for stage_name, stage_config in stages.items():
        print(f"\n  {stage_name}:")
        print(f"    - 名称: {stage_config.name}")
        print(f"    - 训练步数: {stage_config.total_timesteps}")
        print(f"    - 学习率: {stage_config.learning_rate}")
        print(f"    - 猎人数: {stage_config.n_hunters}")
        print(f"    - 猎物数: {stage_config.n_prey}")

    print("\n✅ TrainingConfig测试通过")


def test_hpo_rewards_have_chase_streak():
    """测试HPO奖励函数是否包含持续追击/逃跑加成"""
    print("\n=== 测试2: HPO奖励函数完整性 ===")

    # 测试Stage1HunterRewardHPO
    hunter_reward = Stage1HunterRewardHPO(total_steps=50000, enable_hpo=True)

    assert hasattr(hunter_reward, 'chase_streak'), "❌ Stage1HunterRewardHPO缺少chase_streak属性"
    assert hasattr(hunter_reward, 'max_chase_multiplier'), "❌ Stage1HunterRewardHPO缺少max_chase_multiplier属性"
    assert hasattr(hunter_reward, 'chase_buildup_steps'), "❌ Stage1HunterRewardHPO缺少chase_buildup_steps属性"
    assert hasattr(hunter_reward, 'hpo_enhancer'), "❌ Stage1HunterRewardHPO缺少hpo_enhancer属性"

    print(f"✓ Stage1HunterRewardHPO 包含:")
    print(f"  - chase_streak: {hunter_reward.chase_streak}")
    print(f"  - max_chase_multiplier: {hunter_reward.max_chase_multiplier}")
    print(f"  - chase_buildup_steps: {hunter_reward.chase_buildup_steps}")
    print(f"  - hpo_enhancer: {hunter_reward.hpo_enhancer is not None}")

    # 测试Stage3PreyRewardHPO
    prey_reward = Stage3PreyRewardHPO(total_steps=50000, enable_hpo=True)

    assert hasattr(prey_reward, 'escape_streak'), "❌ Stage3PreyRewardHPO缺少escape_streak属性"
    assert hasattr(prey_reward, 'max_escape_multiplier'), "❌ Stage3PreyRewardHPO缺少max_escape_multiplier属性"
    assert hasattr(prey_reward, 'escape_buildup_steps'), "❌ Stage3PreyRewardHPO缺少escape_buildup_steps属性"
    assert hasattr(prey_reward, 'hpo_enhancer'), "❌ Stage3PreyRewardHPO缺少hpo_enhancer属性"

    print(f"\n✓ Stage3PreyRewardHPO 包含:")
    print(f"  - escape_streak: {prey_reward.escape_streak}")
    print(f"  - max_escape_multiplier: {prey_reward.max_escape_multiplier}")
    print(f"  - escape_buildup_steps: {prey_reward.escape_buildup_steps}")
    print(f"  - hpo_enhancer: {prey_reward.hpo_enhancer is not None}")

    print("\n✅ HPO奖励函数完整性测试通过")


def test_standard_env_creation():
    """测试标准环境创建"""
    print("\n=== 测试3: 标准环境创建 ===")

    agent_config = AgentConfig()
    env_config = EnvConfig()

    stage_config = TrainingConfig.get_stage_config("stage1")

    base_env = EnhancedEcoMARLEnv(
        agent_config=agent_config,
        env_config=env_config,
        n_hunters=stage_config.n_hunters,
        n_prey=stage_config.n_prey,
        max_steps=1000,
        use_v2_rewards=True,
    )

    env = CurriculumEcoMARLEnv(
        base_env=base_env,
        stage="stage1",
    )

    print(f"✓ 创建标准环境成功")
    print(f"  - 观察空间: {env.observation_space.shape}")
    print(f"  - 动作空间: {env.action_space.shape}")
    print(f"  - 阶段: stage1")

    # 测试重置
    obs, info = env.reset()
    print(f"  - 重置成功, 观察维度: {obs.shape}")

    env.close()
    print("✅ 标准环境创建测试通过")


def test_hpo_env_creation():
    """测试HPO环境创建"""
    print("\n=== 测试4: HPO环境创建 ===")

    agent_config = AgentConfig()
    env_config = EnvConfig()

    stage_config = TrainingConfig.get_stage_config("stage1")

    base_env = EnhancedEcoMARLEnv(
        agent_config=agent_config,
        env_config=env_config,
        n_hunters=stage_config.n_hunters,
        n_prey=stage_config.n_prey,
        max_steps=1000,
        use_v2_rewards=True,
    )

    env = CurriculumEcoMARLEnvHPO(
        base_env=base_env,
        stage="stage1",
        enable_hpo=True,
        total_steps=stage_config.total_timesteps,
    )

    print(f"✓ 创建HPO环境成功")
    print(f"  - 观察空间: {env.observation_space.shape}")
    print(f"  - 动作空间: {env.action_space.shape}")
    print(f"  - 阶段: stage1")
    print(f"  - HPO启用: {env.enable_hpo}")

    # 测试重置
    obs, info = env.reset()
    print(f"  - 重置成功, 观察维度: {obs.shape}")

    # 检查HPO增强器
    if hasattr(env, 'hpo_enhancer') and env.hpo_enhancer:
        stats = env.get_hpo_stats()
        print(f"  - HPO统计: {stats is not None}")

    env.close()
    print("✅ HPO环境创建测试通过")


def test_agent_config_parameters():
    """测试AgentConfig参数一致性"""
    print("\n=== 测试5: AgentConfig参数一致性 ===")

    config = AgentConfig()

    # 验证关键参数
    expected_params = {
        'HUNTER_SPEED_MAX': 50.0,
        'HUNTER_ANGULAR_VELOCITY_MAX': 0.15,
        'PREY_SPEED_MAX': 45.0,
        'PREY_ANGULAR_VELOCITY_MAX': 0.18,
    }

    all_pass = True
    for param, expected_value in expected_params.items():
        actual_value = getattr(config, param)
        if actual_value == expected_value:
            print(f"  ✓ {param}: {actual_value}")
        else:
            print(f"  ❌ {param}: 期望{expected_value}, 实际{actual_value}")
            all_pass = False

    if all_pass:
        print("✅ AgentConfig参数一致性测试通过")
    else:
        print("❌ AgentConfig参数一致性测试失败")
        raise AssertionError("AgentConfig参数不一致")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("统一训练脚本测试套件")
    print("=" * 80)

    try:
        test_training_config()
        test_hpo_rewards_have_chase_streak()
        test_standard_env_creation()
        test_hpo_env_creation()
        test_agent_config_parameters()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        print("\n可以安全使用新的train_curriculum.py:")
        print("  python train_curriculum.py --stage stage1          # 标准模式")
        print("  python train_curriculum.py --stage stage1 --enable_hpo  # HPO模式")
        print()

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
