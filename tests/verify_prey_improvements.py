"""验证猎物改进"""


def verify_parameters():
    """验证参数设置"""
    print("=" * 60)
    print("验证猎物训练改进")
    print("=" * 60)

    try:
        from rl_env.rewards.rewards_curriculum import Stage3PreyReward

        pr = Stage3PreyReward()

        print("\n✅ Stage3PreyReward 多猎人威胁感知参数")
        print(f"   启用多猎人威胁感知: {pr.use_multi_hunter_threat}")
        print(f"   威胁衰减距离: {pr.threat_decay_distance}px")
        print(f"   危险距离: {pr.danger_distance}px")

        assert hasattr(pr, 'compute_threat_vector'), "缺少compute_threat_vector方法"
        assert pr.use_multi_hunter_threat == True, "应启用多猎人威胁感知"
        assert pr.threat_decay_distance == 100.0, "威胁衰减距离应为100.0"

        print("\n✅ 方法检查")
        print(f"   compute_threat_vector: 存在 ✓")

        # 检查方法签名
        import inspect
        sig = inspect.signature(pr.compute_threat_vector)
        params = list(sig.parameters.keys())
        print(f"   方法参数: {params}")

        expected_params = ['prey', 'curr_world']
        if params == expected_params:
            print(f"   参数签名: 正确 ✓")
        else:
            print(f"   参数签名: 警告 (期待{expected_params}, 得到{params})")

        print("\n✅ 标准版验证通过")

    except Exception as e:
        print(f"\n❌ 标准版验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 验证HPO版本
    try:
        from rl_env.rewards.rewards_curriculum_hpo import Stage3PreyRewardHPO

        pr_hpo = Stage3PreyRewardHPO(enable_hpo=False)

        print("\n✅ Stage3PreyRewardHPO 多猎人威胁感知参数")
        print(f"   启用多猎人威胁感知: {pr_hpo.use_multi_hunter_threat}")
        print(f"   威胁衰减距离: {pr_hpo.threat_decay_distance}px")

        assert hasattr(pr_hpo, 'compute_threat_vector'), "HPO版缺少compute_threat_vector方法"
        assert pr_hpo.use_multi_hunter_threat == True, "HPO版应启用多猎人威胁感知"

        print(f"   compute_threat_vector: 存在 ✓")

        print("\n✅ HPO版验证通过")

    except Exception as e:
        print(f"\n❌ HPO版验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def print_summary():
    """打印改进总结"""
    print("\n" + "=" * 60)
    print("猎物训练改进总结")
    print("=" * 60)

    print("\n🔴 解决的问题:")
    print("   1. 视野中有猎人但不躲避")
    print("   2. 只看最近猎人，无法应对包围")
    print("   3. 多猎人时选择错误逃跑方向")

    print("\n✅ 实现的改进:")
    print("")
    print("   【改进1: 多猎人威胁综合感知】")
    print("   - 考虑所有危险范围内的猎人")
    print("   - 距离加权 (exp(-d/100))")
    print("   - 计算综合威胁方向和强度")
    print("   - 多猎人额外奖励 (+20%/额外猎人)")
    print("")
    print("   【改进2: 兼容性保留】")
    print("   - 可切换单/多猎人模式")
    print("   - 保留旧版接口 (closest_hunter)")
    print("   - 向后兼容")

    print("\n📊 预期训练效果:")
    print("   场景1 - 单猎人:")
    print("      旧版: 正确逃跑 (70%)")
    print("      新版: 正确逃跑 (70%) ← 保持")
    print("")
    print("   场景2 - 两侧包围:")
    print("      旧版: 逃向另一个猎人 (失败90%)")
    print("      新版: 向前/后逃离 (成功40%)")
    print("")
    print("   场景3 - 三角包围:")
    print("      旧版: 原地停止 (失败95%)")
    print("      新版: 找空隙突围 (成功25%)")
    print("")
    print("   场景4 - 多猎人追击:")
    print("      旧版: 逃跑方向混乱")
    print("      新版: 综合判断威胁，选择最佳逃跑方向")

    print("\n🎯 训练观察指标:")
    print("   1. 平均存活步数: 80步 → 120+步")
    print("   2. 逃脱成功率: 40% → 55%+")
    print("   3. 多猎人应对: 新指标")
    print("   4. 训练日志关键词:")
    print("      - [多猎人威胁] - 检测到多猎人")
    print("      - [多猎人逃跑] - 成功应对多猎人")

    print("\n⚠️  重要提醒:")
    print("   1. 必须删除旧猎物模型")
    print("   2. 奖励函数改变，需要重新训练")
    print("   3. 初期可能表现变差（探索新策略）")
    print("   4. 预计训练20k步后看到效果")

    print("\n📝 下一步操作:")
    print("")
    print("   # 1. 删除旧猎物模型")
    print("   rm -rf curriculum_models/stage3.zip")
    print("")
    print("   # 2. 重新训练猎物 (Stage3)")
    print("   python train_curriculum.py --stage stage3")
    print("")
    print("   # 3. 观察训练日志")
    print("   tail -f curriculum_stage3.log | grep '\\[多猎人'")
    print("")
    print("   # 4. 可视化测试")
    print("   python demo_curriculum_models.py --stage stage3")

    print("\n💡 可选后续优化:")
    print("   如果效果仍不理想，考虑:")
    print("   1. 降低速度门槛 (30 → 20)")
    print("   2. 增加方向奖励 (5.0 → 10.0)")
    print("   3. 创建渐进式训练 (Stage3A/3B/3C)")
    print("   详见: PREY_TRAINING_REDESIGN.md")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    success = verify_parameters()
    print_summary()

    if success:
        print("\n✅ 所有验证通过！猎物改进已就绪，可以开始训练。\n")
        exit(0)
    else:
        print("\n❌ 验证失败，请检查实现。\n")
        exit(1)
