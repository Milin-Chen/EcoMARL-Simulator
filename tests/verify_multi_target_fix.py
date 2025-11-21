"""验证多目标摇摆/停顿修复"""

def verify_parameters():
    """验证参数设置"""
    print("=" * 60)
    print("验证多目标摇摆/停顿修复参数")
    print("=" * 60)

    try:
        from rl_env.rewards.rewards_curriculum import (
            Stage1HunterReward,
            Stage2HunterReward,
            Stage3PreyReward
        )

        hr1 = Stage1HunterReward()
        hr2 = Stage2HunterReward()
        pr = Stage3PreyReward()

        print("\n✅ Stage1HunterReward 参数")
        print(f"   锁定步数: {hr1.min_lock_steps}")
        print(f"   切换惩罚: {hr1.target_switch_penalty}")
        print(f"   抖动惩罚: {hr1.jitter_penalty}")
        print(f"   抖动阈值: {hr1.jitter_radius_threshold}px")
        print(f"   历史长度: {hr1.position_history_length}步")

        assert hr1.min_lock_steps == 8, "锁定步数应为8"
        assert hr1.target_switch_penalty == -15.0, "切换惩罚应为-15.0"
        assert hr1.jitter_penalty == -12.0, "抖动惩罚应为-12.0"
        assert hr1.jitter_radius_threshold == 20.0, "抖动阈值应为20.0"

        print("\n✅ Stage2HunterReward 参数")
        print(f"   锁定步数: {hr2.min_lock_steps}")
        print(f"   切换惩罚: {hr2.target_switch_penalty}")
        print(f"   抖动惩罚: {hr2.jitter_penalty}")
        print(f"   抖动阈值: {hr2.jitter_radius_threshold}px")

        assert hr2.min_lock_steps == 8, "锁定步数应为8"
        assert hr2.target_switch_penalty == -15.0, "切换惩罚应为-15.0"

        print("\n✅ Stage3PreyReward 参数")
        print(f"   抖动惩罚: {pr.jitter_penalty}")
        print(f"   抖动阈值: {pr.jitter_radius_threshold}px")
        print(f"   聚集冲突惩罚: {pr.herd_escape_conflict_penalty}")
        print(f"   危险距离: {pr.dangerous_herd_distance}px")
        print(f"   历史长度: {pr.position_history_length}步")

        assert pr.jitter_penalty == -15.0, "抖动惩罚应为-15.0"
        assert pr.jitter_radius_threshold == 15.0, "抖动阈值应为15.0"
        assert pr.herd_escape_conflict_penalty == -10.0, "聚集冲突惩罚应为-10.0"
        assert pr.dangerous_herd_distance == 200.0, "危险距离应为200.0"

        print("\n" + "=" * 60)
        print("✅ 标准版参数验证通过")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 标准版验证失败: {e}")
        return False

    # 验证HPO版本
    try:
        from rl_env.rewards.rewards_curriculum_hpo import (
            Stage1HunterRewardHPO,
            Stage3PreyRewardHPO
        )

        hr_hpo = Stage1HunterRewardHPO(enable_hpo=False)
        pr_hpo = Stage3PreyRewardHPO(enable_hpo=False)

        print("\n✅ Stage1HunterRewardHPO 参数")
        print(f"   锁定步数: {hr_hpo.min_lock_steps}")
        print(f"   切换惩罚: {hr_hpo.target_switch_penalty}")
        print(f"   抖动惩罚: {hr_hpo.jitter_penalty}")
        print(f"   抖动阈值: {hr_hpo.jitter_radius_threshold}px")

        assert hr_hpo.min_lock_steps == 8, "HPO锁定步数应为8"
        assert hr_hpo.target_switch_penalty == -15.0, "HPO切换惩罚应为-15.0"
        assert hr_hpo.jitter_penalty == -12.0, "HPO抖动惩罚应为-12.0"

        print("\n✅ Stage3PreyRewardHPO 参数")
        print(f"   抖动惩罚: {pr_hpo.jitter_penalty}")
        print(f"   抖动阈值: {pr_hpo.jitter_radius_threshold}px")
        print(f"   聚集冲突惩罚: {pr_hpo.herd_escape_conflict_penalty}")
        print(f"   危险距离: {pr_hpo.dangerous_herd_distance}px")

        assert pr_hpo.jitter_penalty == -15.0, "HPO抖动惩罚应为-15.0"
        assert pr_hpo.herd_escape_conflict_penalty == -10.0, "HPO聚集冲突惩罚应为-10.0"

        print("\n" + "=" * 60)
        print("✅ HPO版参数验证通过")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ HPO版验证失败: {e}")
        return False

    return True


def print_fix_summary():
    """打印修复总结"""
    print("\n" + "=" * 60)
    print("修复内容总结")
    print("=" * 60)

    print("\n🎯 修复的问题:")
    print("   1. 猎人视野中出现多个猎物时原地摇摆")
    print("   2. 猎物视野中出现多个同类/敌人时原地停止")

    print("\n✅ 实现的修复:")
    print("   1. 目标锁定机制 (猎人)")
    print("      - 强制锁定目标至少8步")
    print("      - 目标切换惩罚: -15.0")
    print("")
    print("   2. 位置抖动检测 (猎人+猎物)")
    print("      - 追踪5步位置历史")
    print("      - 活动半径<阈值 → 惩罚")
    print("      - 猎人: -12.0 (半径<20px)")
    print("      - 猎物: -15.0 (半径<15px)")
    print("")
    print("   3. 集群-逃跑冲突检测 (猎物)")
    print("      - 安全时聚集 → 奖励")
    print("      - 危险时聚集 → 惩罚-10.0")
    print("      - 危险距离: 猎人<200px")

    print("\n📊 预期效果:")
    print("   - 猎人专注追击单个目标")
    print("   - 猎人持续高速移动")
    print("   - 猎物危险时优先逃跑")
    print("   - 猎物高速移动逃离")
    print("   - 原地摇摆/停顿大幅减少")

    print("\n⚠️  下一步:")
    print("   1. 删除旧模型: rm -rf curriculum_models/stage*.zip")
    print("   2. 重新训练: python train_curriculum.py --stage <stage>")
    print("   3. 观察指标: 目标切换次数、抖动次数、成功率")
    print("   4. 可视化测试: python demo_curriculum_models.py")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    success = verify_parameters()
    print_fix_summary()

    if success:
        print("\n✅ 所有验证通过！可以开始重新训练模型。\n")
        exit(0)
    else:
        print("\n❌ 验证失败，请检查修改。\n")
        exit(1)
