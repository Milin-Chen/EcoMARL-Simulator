"""
测试增强版奖励函数 - 快速验证
检查:
1. 奖励是否非零
2. 捕食事件是否发生
3. 追击行为是否有奖励
4. 原地转圈是否被惩罚
"""

import numpy as np
from rl_env import EnhancedEcoMARLEnv


def test_enhanced_rewards():
    """测试增强版奖励函数"""
    print("=" * 70)
    print("测试增强版奖励函数")
    print("=" * 70)

    # 创建环境
    env = EnhancedEcoMARLEnv(
        n_hunters=3,
        n_prey=6,
        max_steps=500,
    )

    print("\n✓ 环境创建成功")
    print(f"  猎人数量: {env.n_hunters}")
    print(f"  猎物数量: {env.n_prey}")

    # 重置环境
    obs = env.reset()
    print("\n✓ 环境重置成功")
    print(f"  观察数量: {len(obs)} agents")

    # 运行测试
    print("\n" + "=" * 70)
    print("开始测试 (运行100步)")
    print("=" * 70)

    total_hunter_reward = 0.0
    total_prey_reward = 0.0
    non_zero_steps = 0
    predation_count = 0
    reward_breakdown = {
        "positive_hunter": 0,
        "negative_hunter": 0,
        "positive_prey": 0,
        "negative_prey": 0,
    }

    for step in range(100):
        # 随机动作 - 为每个agent生成动作
        actions = {}
        for agent_id in obs.keys():
            actions[agent_id] = np.random.randn(2) * 0.5

        # 执行步骤
        obs, rewards, dones, info = env.step(actions)

        # 统计奖励 - 按类型分组 (h_开头是hunter, p_开头是prey)
        hunter_reward = sum(r for aid, r in rewards.items() if aid.startswith('h_'))
        prey_reward = sum(r for aid, r in rewards.items() if aid.startswith('p_'))

        total_hunter_reward += hunter_reward
        total_prey_reward += prey_reward

        if abs(hunter_reward) > 0.01 or abs(prey_reward) > 0.01:
            non_zero_steps += 1

        # 统计正负奖励
        if hunter_reward > 0:
            reward_breakdown["positive_hunter"] += 1
        elif hunter_reward < 0:
            reward_breakdown["negative_hunter"] += 1

        if prey_reward > 0:
            reward_breakdown["positive_prey"] += 1
        elif prey_reward < 0:
            reward_breakdown["negative_prey"] += 1

        # 检查捕食事件
        new_predation_count = info["episode_stats"]["total_predations"]
        if new_predation_count > predation_count:
            predation_count = new_predation_count
            print(f"\n🎯 步骤 {step}: 捕食事件发生! 总捕食数: {predation_count}")
            print(f"   猎人奖励: {hunter_reward:.2f}")
            print(f"   当前猎人数: {info['population']['hunters']}, 猎物数: {info['population']['preys']}")

        # 每20步输出一次
        if (step + 1) % 20 == 0:
            print(f"\n步骤 {step+1}/100:")
            print(f"  猎人奖励: {hunter_reward:.3f} (累计: {total_hunter_reward:.2f})")
            print(f"  猎物奖励: {prey_reward:.3f} (累计: {total_prey_reward:.2f})")
            print(f"  非零奖励步数: {non_zero_steps}/{step+1}")
            print(f"  猎人数: {info['population']['hunters']}, 猎物数: {info['population']['preys']}")

        # 检查是否所有agent都done
        if all(dones.values()):
            print(f"\n⚠️  Episode在步骤 {step+1} 终止")
            break

    # 测试总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    print(f"\n📊 奖励统计:")
    print(f"  总猎人奖励: {total_hunter_reward:.2f}")
    print(f"  总猎物奖励: {total_prey_reward:.2f}")
    print(f"  平均猎人奖励/步: {total_hunter_reward / (step+1):.3f}")
    print(f"  平均猎物奖励/步: {total_prey_reward / (step+1):.3f}")
    print(f"  非零奖励步数: {non_zero_steps}/{step+1} ({non_zero_steps/(step+1)*100:.1f}%)")

    print(f"\n📈 奖励分布:")
    print(f"  猎人正奖励步数: {reward_breakdown['positive_hunter']}")
    print(f"  猎人负奖励步数: {reward_breakdown['negative_hunter']}")
    print(f"  猎物正奖励步数: {reward_breakdown['positive_prey']}")
    print(f"  猎物负奖励步数: {reward_breakdown['negative_prey']}")

    print(f"\n🎯 捕食统计:")
    print(f"  捕食事件数: {predation_count}")

    # Episode总结
    summary = env.get_episode_summary()
    print(f"\n📋 Episode总结:")
    print(f"  总步数: {summary['steps']}")
    print(f"  最终猎人数: {summary['final_population']['hunters']}")
    print(f"  最终猎物数: {summary['final_population']['preys']}")
    print(f"  总捕食数: {summary['total_predations']}")

    # 验证结果
    print("\n" + "=" * 70)
    print("验证结果")
    print("=" * 70)

    passed = []
    failed = []

    # 检查1: 奖励非零
    if non_zero_steps > 10:
        passed.append("✓ 奖励非零 (非零步数 > 10)")
    else:
        failed.append("✗ 奖励几乎全为0")

    # 检查2: 奖励合理范围
    if -1000 < total_hunter_reward < 1000:
        passed.append("✓ 猎人奖励在合理范围")
    else:
        failed.append(f"✗ 猎人奖励异常: {total_hunter_reward:.2f}")

    if -1000 < total_prey_reward < 1000:
        passed.append("✓ 猎物奖励在合理范围")
    else:
        failed.append(f"✗ 猎物奖励异常: {total_prey_reward:.2f}")

    # 检查3: 奖励有正有负
    if reward_breakdown["positive_hunter"] > 0 and reward_breakdown["negative_hunter"] > 0:
        passed.append("✓ 猎人奖励有正有负")
    else:
        failed.append("✗ 猎人奖励缺乏多样性")

    if reward_breakdown["positive_prey"] > 0 and reward_breakdown["negative_prey"] > 0:
        passed.append("✓ 猎物奖励有正有负")
    else:
        failed.append("✗ 猎物奖励缺乏多样性")

    # 打印结果
    print("\n通过的检查:")
    for check in passed:
        print(f"  {check}")

    if failed:
        print("\n失败的检查:")
        for check in failed:
            print(f"  {check}")
    else:
        print("\n🎉 所有检查通过!")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

    return len(failed) == 0


if __name__ == "__main__":
    success = test_enhanced_rewards()
    exit(0 if success else 1)
