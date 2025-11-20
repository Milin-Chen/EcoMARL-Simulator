"""评估和对比不同检查点的性能"""

import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from rl_env import EcoMARLEnv, SharedPolicy
from config import EnvConfig, AgentConfig


def evaluate_checkpoint(
    model_path: str,
    n_episodes: int = 10,
    n_hunters: int = 6,
    n_prey: int = 18,
    max_steps: int = 1000,
    device: str = "cpu",
) -> Dict:
    """评估单个检查点"""
    print(f"\n评估检查点: {model_path}")

    # 创建环境
    env = EcoMARLEnv(
        env_config=EnvConfig(),
        agent_config=AgentConfig(),
        n_hunters=n_hunters,
        n_prey=n_prey,
        use_parallel=True,
    )

    # 加载模型
    obs_info = env.get_observation_space()
    policy = SharedPolicy(obs_dim=obs_info["observation_dim"], action_dim=2, device=device)
    policy.load(model_path)

    # 统计数据
    episode_stats = []

    for episode in range(n_episodes):
        observations = env.reset()
        episode_reward = {"hunter": [], "prey": []}
        episode_survival = {"hunter": 0, "prey": 0}
        episode_predations = 0
        episode_births = 0

        for step in range(max_steps):
            # 准备观察
            obs_dict = {}
            for agent_id, obs in observations.items():
                entity = next((e for e in env.world.entities if e.id == agent_id), None)
                if entity:
                    obs_dict[agent_id] = (obs, entity.type)

            # 获取动作（确定性）
            actions_dict, _, _ = policy.get_actions_batch(obs_dict, deterministic=True)

            # 执行动作
            observations, rewards, dones, info = env.step(actions_dict)

            # 记录奖励
            for agent_id, reward in rewards.items():
                entity = next((e for e in env.world.entities if e.id == agent_id), None)
                if entity:
                    episode_reward[entity.type].append(reward)

            # 记录事件
            for event in env.world.events:
                if event.type == "predation":
                    episode_predations += 1
                elif event.type == "breed":
                    episode_births += 1

            # 如果所有实体都死亡，提前结束
            if len(observations) == 0:
                break

        # 记录最终存活数量
        stats = env.simulator.get_stats()
        episode_survival["hunter"] = stats.get("hunters", 0)
        episode_survival["prey"] = stats.get("preys", 0)

        # 计算平均奖励
        hunter_avg_reward = (
            np.mean(episode_reward["hunter"]) if episode_reward["hunter"] else 0.0
        )
        prey_avg_reward = (
            np.mean(episode_reward["prey"]) if episode_reward["prey"] else 0.0
        )

        episode_stats.append(
            {
                "hunter_reward": hunter_avg_reward,
                "prey_reward": prey_avg_reward,
                "hunter_survival": episode_survival["hunter"],
                "prey_survival": episode_survival["prey"],
                "predations": episode_predations,
                "births": episode_births,
                "total_steps": step + 1,
            }
        )

        print(
            f"  Episode {episode + 1}/{n_episodes}: "
            f"猎人奖励={hunter_avg_reward:.2f}, "
            f"猎物奖励={prey_avg_reward:.2f}, "
            f"存活={episode_survival['hunter']}H/{episode_survival['prey']}P"
        )

    env.close()

    # 计算汇总统计
    summary = {
        "hunter_reward_mean": np.mean([s["hunter_reward"] for s in episode_stats]),
        "hunter_reward_std": np.std([s["hunter_reward"] for s in episode_stats]),
        "prey_reward_mean": np.mean([s["prey_reward"] for s in episode_stats]),
        "prey_reward_std": np.std([s["prey_reward"] for s in episode_stats]),
        "hunter_survival_mean": np.mean([s["hunter_survival"] for s in episode_stats]),
        "prey_survival_mean": np.mean([s["prey_survival"] for s in episode_stats]),
        "predations_mean": np.mean([s["predations"] for s in episode_stats]),
        "births_mean": np.mean([s["births"] for s in episode_stats]),
        "steps_mean": np.mean([s["total_steps"] for s in episode_stats]),
    }

    print(f"\n检查点汇总:")
    print(f"  猎人平均奖励: {summary['hunter_reward_mean']:.2f} ± {summary['hunter_reward_std']:.2f}")
    print(f"  猎物平均奖励: {summary['prey_reward_mean']:.2f} ± {summary['prey_reward_std']:.2f}")
    print(f"  平均存活: {summary['hunter_survival_mean']:.1f}H / {summary['prey_survival_mean']:.1f}P")
    print(f"  平均捕食次数: {summary['predations_mean']:.1f}")
    print(f"  平均繁殖次数: {summary['births_mean']:.1f}")

    return summary


def compare_all_checkpoints(
    run_dir: str,
    n_episodes: int = 10,
    output_path: str = None,
    device: str = "cpu",
):
    """对比所有检查点的性能"""
    print("=" * 70)
    print("检查点性能评估与对比")
    print("=" * 70)

    # 查找所有模型检查点
    model_files = sorted(Path(run_dir).glob("model_step_*.pt"))

    if not model_files:
        print(f"错误: 在 {run_dir} 中没有找到模型检查点")
        return

    print(f"\n找到 {len(model_files)} 个检查点")

    # 评估每个检查点
    results = {}
    for model_file in model_files:
        step = int(model_file.stem.split("_")[-1])
        summary = evaluate_checkpoint(
            str(model_file), n_episodes=n_episodes, device=device
        )
        results[step] = summary

    # 可视化对比
    plot_checkpoint_comparison(results, output_path)

    # 打印对比表
    print("\n" + "=" * 70)
    print("检查点对比表")
    print("=" * 70)
    print(
        f"{'步数':>8} | {'猎人奖励':>10} | {'猎物奖励':>10} | "
        f"{'猎人存活':>8} | {'猎物存活':>8} | {'捕食':>6} | {'繁殖':>6}"
    )
    print("-" * 70)

    for step in sorted(results.keys()):
        r = results[step]
        print(
            f"{step:8d} | {r['hunter_reward_mean']:10.2f} | {r['prey_reward_mean']:10.2f} | "
            f"{r['hunter_survival_mean']:8.1f} | {r['prey_survival_mean']:8.1f} | "
            f"{r['predations_mean']:6.1f} | {r['births_mean']:6.1f}"
        )


def plot_checkpoint_comparison(results: Dict, save_path: str = None):
    """绘制检查点对比图"""
    steps = sorted(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("检查点性能对比", fontsize=16, fontweight="bold")

    # 1. 猎人奖励
    ax = axes[0, 0]
    hunter_rewards = [results[s]["hunter_reward_mean"] for s in steps]
    hunter_stds = [results[s]["hunter_reward_std"] for s in steps]
    ax.errorbar(steps, hunter_rewards, yerr=hunter_stds, marker="o", color="red", linewidth=2, capsize=5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均奖励")
    ax.set_title("猎人奖励")
    ax.grid(True, alpha=0.3)

    # 2. 猎物奖励
    ax = axes[0, 1]
    prey_rewards = [results[s]["prey_reward_mean"] for s in steps]
    prey_stds = [results[s]["prey_reward_std"] for s in steps]
    ax.errorbar(steps, prey_rewards, yerr=prey_stds, marker="o", color="green", linewidth=2, capsize=5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均奖励")
    ax.set_title("猎物奖励")
    ax.grid(True, alpha=0.3)

    # 3. 存活数量
    ax = axes[0, 2]
    hunter_survival = [results[s]["hunter_survival_mean"] for s in steps]
    prey_survival = [results[s]["prey_survival_mean"] for s in steps]
    ax.plot(steps, hunter_survival, marker="o", label="猎人", color="red", linewidth=2)
    ax.plot(steps, prey_survival, marker="o", label="猎物", color="green", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均存活数量")
    ax.set_title("存活数量")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 捕食次数
    ax = axes[1, 0]
    predations = [results[s]["predations_mean"] for s in steps]
    ax.plot(steps, predations, marker="o", color="orange", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均捕食次数")
    ax.set_title("捕食效率")
    ax.grid(True, alpha=0.3)

    # 5. 繁殖次数
    ax = axes[1, 1]
    births = [results[s]["births_mean"] for s in steps]
    ax.plot(steps, births, marker="o", color="purple", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均繁殖次数")
    ax.set_title("繁殖效率")
    ax.grid(True, alpha=0.3)

    # 6. 平均步数
    ax = axes[1, 2]
    avg_steps = [results[s]["steps_mean"] for s in steps]
    ax.plot(steps, avg_steps, marker="o", color="blue", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均运行步数")
    ax.set_title("Episode长度")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n对比图已保存到: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估和对比检查点")
    parser.add_argument("--run_dir", type=str, required=True, help="训练运行目录")
    parser.add_argument("--checkpoint", type=str, help="单个检查点路径（可选）")
    parser.add_argument("--n_episodes", type=int, default=10, help="评估episode数量")
    parser.add_argument("--output", type=str, help="输出图表路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备")

    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("错误: 需要安装 matplotlib")
        print("安装命令: pip install matplotlib")
        exit(1)

    if args.checkpoint:
        # 评估单个检查点
        evaluate_checkpoint(
            args.checkpoint, n_episodes=args.n_episodes, device=args.device
        )
    else:
        # 对比所有检查点
        output_path = args.output or os.path.join(
            args.run_dir, "visualizations", "checkpoint_performance.png"
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        compare_all_checkpoints(
            args.run_dir, n_episodes=args.n_episodes, output_path=output_path, device=args.device
        )
