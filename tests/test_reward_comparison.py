"""
对比增强版V1和V2奖励函数

测试随机运动下的奖励差异
"""

import numpy as np
from rl_env import (
    EnhancedEcoMARLEnv,
    EnhancedRewardFunction,
    EnhancedRewardFunctionV2,
)


def test_random_movement_rewards(n_steps=100):
    """测试随机运动下的奖励"""

    # 创建环境
    env = EnhancedEcoMARLEnv(n_hunters=3, n_prey=6)

    # 创建两个奖励函数
    reward_v1 = EnhancedRewardFunction()
    reward_v2 = EnhancedRewardFunctionV2()

    # 记录奖励
    v1_rewards = {"hunter": [], "prey": []}
    v2_rewards = {"hunter": [], "prey": []}

    print("="*70)
    print("测试随机运动下的奖励对比 - V1 vs V2")
    print("="*70)
    print(f"\n运行 {n_steps} 步随机动作...\n")

    observations = env.reset()

    for step in range(n_steps):
        # 保存前一个世界状态
        prev_world = env.world

        # 随机动作 (模拟完全随机的策略)
        actions = {}
        for agent_id in observations.keys():
            # 随机线性和角加速度: [-1, 1]
            actions[agent_id] = np.random.uniform(-1, 1, size=2)

        # Step环境
        observations, _, dones, _ = env.step(actions)

        # 计算两个版本的奖励
        curr_world = env.world
        for entity in curr_world.entities:
            prev_entity = next(
                (e for e in prev_world.entities if e.id == entity.id), None
            )
            if prev_entity is None:
                continue

            # V1奖励 (旧接口:不需要prev_entity)
            r_v1 = reward_v1.compute_reward(entity, prev_world, curr_world)
            v1_rewards[entity.type].append(r_v1)

            # V2奖励 (新接口:需要prev_entity)
            r_v2 = reward_v2.compute_reward(entity, prev_entity, prev_world, curr_world)
            v2_rewards[entity.type].append(r_v2)

        # 每20步打印一次
        if (step + 1) % 20 == 0:
            print(f"步骤 {step + 1}/{n_steps}:")

            # V1统计
            v1_h_mean = np.mean(v1_rewards["hunter"][-20*3:]) if v1_rewards["hunter"] else 0
            v1_p_mean = np.mean(v1_rewards["prey"][-20*6:]) if v1_rewards["prey"] else 0

            # V2统计
            v2_h_mean = np.mean(v2_rewards["hunter"][-20*3:]) if v2_rewards["hunter"] else 0
            v2_p_mean = np.mean(v2_rewards["prey"][-20*6:]) if v2_rewards["prey"] else 0

            print(f"  V1 - 猎人: {v1_h_mean:7.3f} | 猎物: {v1_p_mean:7.3f}")
            print(f"  V2 - 猎人: {v2_h_mean:7.3f} | 猎物: {v2_p_mean:7.3f}")
            print()

    # 最终统计
    print("="*70)
    print("最终统计结果")
    print("="*70)

    print("\n【V1 - 增强版原始】:")
    print(f"  猎人平均奖励: {np.mean(v1_rewards['hunter']):.3f} ± {np.std(v1_rewards['hunter']):.3f}")
    print(f"  猎物平均奖励: {np.mean(v1_rewards['prey']):.3f} ± {np.std(v1_rewards['prey']):.3f}")
    print(f"  猎人正奖励比例: {sum(1 for r in v1_rewards['hunter'] if r > 0) / len(v1_rewards['hunter']) * 100:.1f}%")
    print(f"  猎物正奖励比例: {sum(1 for r in v1_rewards['prey'] if r > 0) / len(v1_rewards['prey']) * 100:.1f}%")

    print("\n【V2 - 意图性检测版】:")
    print(f"  猎人平均奖励: {np.mean(v2_rewards['hunter']):.3f} ± {np.std(v2_rewards['hunter']):.3f}")
    print(f"  猎物平均奖励: {np.mean(v2_rewards['prey']):.3f} ± {np.std(v2_rewards['prey']):.3f}")
    print(f"  猎人正奖励比例: {sum(1 for r in v2_rewards['hunter'] if r > 0) / len(v2_rewards['hunter']) * 100:.1f}%")
    print(f"  猎物正奖励比例: {sum(1 for r in v2_rewards['prey'] if r > 0) / len(v2_rewards['prey']) * 100:.1f}%")

    print("\n【改进幅度】:")
    v1_h_mean = np.mean(v1_rewards['hunter'])
    v2_h_mean = np.mean(v2_rewards['hunter'])
    v1_p_mean = np.mean(v1_rewards['prey'])
    v2_p_mean = np.mean(v2_rewards['prey'])

    if abs(v1_h_mean) > 0.001:
        h_improvement = (v2_h_mean - v1_h_mean) / abs(v1_h_mean) * 100
        print(f"  猎人奖励变化: {h_improvement:+.1f}%")

    if abs(v1_p_mean) > 0.001:
        p_improvement = (v2_p_mean - v1_p_mean) / abs(v1_p_mean) * 100
        print(f"  猎物奖励变化: {p_improvement:+.1f}%")

    print("\n【分析】:")
    if v2_h_mean < v1_h_mean:
        print("  ✓ V2成功降低了随机策略的猎人奖励")
    else:
        print("  ✗ V2未能降低随机策略的猎人奖励")

    if v2_p_mean < v1_p_mean:
        print("  ✓ V2成功降低了随机策略的猎物奖励")
    else:
        print("  ✗ V2未能降低随机策略的猎物奖励")

    # 检查V2是否有更多负奖励
    v1_neg_ratio = sum(1 for r in v1_rewards['hunter'] if r < 0) / len(v1_rewards['hunter'])
    v2_neg_ratio = sum(1 for r in v2_rewards['hunter'] if r < 0) / len(v2_rewards['hunter'])

    if v2_neg_ratio > v1_neg_ratio:
        print(f"  ✓ V2增加了负奖励比例 ({v1_neg_ratio*100:.1f}% → {v2_neg_ratio*100:.1f}%)")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)


if __name__ == "__main__":
    test_random_movement_rewards(n_steps=100)
