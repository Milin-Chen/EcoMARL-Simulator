"""
测试统一配置文件
Test Unified Configuration

验证:
1. TrainingConfig正确加载
2. HPO奖励函数包含持续追击/逃跑加成（代码检查）
3. AgentConfig参数一致性
"""

import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TrainingConfig, AgentConfig, EnvConfig


def test_training_config():
    """测试TrainingConfig加载"""
    print("\n=== 测试1: TrainingConfig加载 ===")

    config = TrainingConfig()
    print(f"✓ PPO参数:")
    print(f"  - n_steps: {config.PPO_N_STEPS}")
    print(f"  - batch_size: {config.PPO_BATCH_SIZE}")
    print(f"  - gamma: {config.PPO_GAMMA}")
    print(f"  - gae_lambda: {config.PPO_GAE_LAMBDA}")

    stages = config.get_stage_configs()
    print(f"\n✓ 课程学习阶段: {list(stages.keys())}")

    for stage_name, stage_config in stages.items():
        print(f"\n  {stage_name}:")
        print(f"    - 名称: {stage_config.name}")
        print(f"    - 训练步数: {stage_config.total_timesteps:,}")
        print(f"    - 学习率: {stage_config.learning_rate}")
        print(f"    - 猎人数: {stage_config.n_hunters}, 猎物数: {stage_config.n_prey}")
        print(f"    - 训练猎人: {stage_config.train_hunters}, 训练猎物: {stage_config.train_prey}")

    print("\n✅ TrainingConfig测试通过")


def test_hpo_rewards_code_check():
    """测试HPO奖励函数代码是否包含追击加成"""
    print("\n=== 测试2: HPO奖励函数代码检查 ===")

    # 读取rewards_curriculum_hpo.py文件
    rewards_hpo_file = PROJECT_ROOT / "rl_env" / "rewards_curriculum_hpo.py"

    with open(rewards_hpo_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查Stage1HunterRewardHPO
    hunter_checks = [
        ('chase_streak', 'Stage1HunterRewardHPO包含chase_streak'),
        ('max_chase_multiplier', 'Stage1HunterRewardHPO包含max_chase_multiplier'),
        ('chase_buildup_steps', 'Stage1HunterRewardHPO包含chase_buildup_steps'),
        ('chase_multiplier = 1.0', 'Stage1HunterRewardHPO计算chase_multiplier'),
    ]

    print("✓ Stage1HunterRewardHPO检查:")
    for keyword, description in hunter_checks:
        if keyword in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description} - 未找到关键词: {keyword}")
            raise AssertionError(f"HPO奖励函数缺少: {keyword}")

    # 检查Stage3PreyRewardHPO
    prey_checks = [
        ('escape_streak', 'Stage3PreyRewardHPO包含escape_streak'),
        ('max_escape_multiplier', 'Stage3PreyRewardHPO包含max_escape_multiplier'),
        ('escape_buildup_steps', 'Stage3PreyRewardHPO包含escape_buildup_steps'),
        ('escape_multiplier = 1.0', 'Stage3PreyRewardHPO计算escape_multiplier'),
    ]

    print("\n✓ Stage3PreyRewardHPO检查:")
    for keyword, description in prey_checks:
        if keyword in content:
            print(f"  ✓ {description}")
        else:
            print(f"  ❌ {description} - 未找到关键词: {keyword}")
            raise AssertionError(f"HPO奖励函数缺少: {keyword}")

    print("\n✅ HPO奖励函数代码检查通过")


def test_agent_config_parameters():
    """测试AgentConfig参数一致性"""
    print("\n=== 测试3: AgentConfig参数一致性 ===")

    config = AgentConfig()

    # 验证关键参数
    expected_params = {
        'HUNTER_SPEED_MAX': 50.0,
        'HUNTER_ANGULAR_VELOCITY_MAX': 0.15,
        'PREY_SPEED_MAX': 45.0,
        'PREY_ANGULAR_VELOCITY_MAX': 0.18,
        'SPEED_DELTA_MAX': 10.0,
        'ANGULAR_DELTA_MAX': 0.2,
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
        raise AssertionError("AgentConfig参数不一致")


def test_env_config():
    """测试EnvConfig"""
    print("\n=== 测试4: EnvConfig ===")

    config = EnvConfig()

    print(f"✓ 世界配置:")
    print(f"  - 窗口大小: {config.WINDOW_WIDTH} x {config.WINDOW_HEIGHT}")
    print(f"  - 最大实体数: {config.MAX_ENTITIES}")
    print(f"  - DT: {config.DT}")

    print(f"\n✓ 能量配置:")
    print(f"  - 猎人基础代谢: {config.ENERGY_BASE_METABOLISM_HUNTER}")
    print(f"  - 猎物基础代谢: {config.ENERGY_BASE_METABOLISM_PREY}")
    print(f"  - 猎人最大能量: {config.ENERGY_MAX_HUNTER}")
    print(f"  - 猎物最大能量: {config.ENERGY_MAX_PREY}")

    print("✅ EnvConfig测试通过")


def test_training_config_stage_consistency():
    """测试阶段配置的一致性"""
    print("\n=== 测试5: 阶段配置一致性 ===")

    config = TrainingConfig()

    # 验证所有阶段都有必需的属性
    required_attrs = [
        'name', 'description', 'n_hunters', 'n_prey',
        'total_timesteps', 'learning_rate', 'prey_behavior',
        'train_hunters', 'train_prey', 'success_criteria'
    ]

    stages = config.get_stage_configs()
    for stage_name, stage_config in stages.items():
        print(f"\n✓ 检查 {stage_name}:")
        for attr in required_attrs:
            if not hasattr(stage_config, attr):
                print(f"  ❌ 缺少属性: {attr}")
                raise AssertionError(f"{stage_name} 缺少必需属性: {attr}")
            print(f"  ✓ {attr}: {getattr(stage_config, attr)}")

    print("\n✅ 阶段配置一致性测试通过")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("统一配置测试套件")
    print("=" * 80)

    try:
        test_training_config()
        test_hpo_rewards_code_check()
        test_agent_config_parameters()
        test_env_config()
        test_training_config_stage_consistency()

        print("\n" + "=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        print("\n✅ 配置文件验证成功:")
        print("  1. ✓ TrainingConfig 正确加载所有阶段配置")
        print("  2. ✓ HPO奖励函数包含持续追击/逃跑加成机制")
        print("  3. ✓ AgentConfig 参数统一 (50.0, 45.0, 0.15, 0.18)")
        print("  4. ✓ EnvConfig 正确配置能量和世界参数")
        print("  5. ✓ 所有阶段配置完整且一致")

        print("\n📝 可以安全使用新的train_curriculum.py:")
        print("  # 标准模式 (默认)")
        print("  python train_curriculum.py --stage stage1")
        print()
        print("  # HPO增强模式")
        print("  python train_curriculum.py --stage stage1 --enable_hpo")
        print()
        print("  # 训练所有阶段")
        print("  python train_curriculum.py --stages stage1 stage2 stage3 stage4")
        print()

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ 测试失败: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
