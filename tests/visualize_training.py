"""训练过程可视化工具"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def load_training_history(history_path: str) -> Dict:
    """加载训练历史"""
    with open(history_path, "r") as f:
        return json.load(f)


def plot_training_curves(history: Dict, save_path: str = None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("训练过程曲线", fontsize=16, fontweight="bold")

    timesteps = history["timesteps"]

    # 1. 平均奖励
    ax = axes[0, 0]
    ax.plot(timesteps, history["hunter_mean_reward"], label="猎人", color="red", linewidth=2)
    ax.plot(timesteps, history["prey_mean_reward"], label="猎物", color="green", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均奖励")
    ax.set_title("平均奖励曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 策略损失
    ax = axes[0, 1]
    ax.plot(timesteps, history["hunter_policy_loss"], label="猎人", color="red", linewidth=2)
    ax.plot(timesteps, history["prey_policy_loss"], label="猎物", color="green", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("策略损失")
    ax.set_title("策略损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 价值损失
    ax = axes[0, 2]
    ax.plot(timesteps, history["hunter_value_loss"], label="猎人", color="red", linewidth=2)
    ax.plot(timesteps, history["prey_value_loss"], label="猎物", color="green", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("价值损失")
    ax.set_title("价值损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 种群数量
    ax = axes[1, 0]
    ax.plot(timesteps, history["n_hunters"], label="猎人", color="red", linewidth=2)
    ax.plot(timesteps, history["n_preys"], label="猎物", color="green", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("数量")
    ax.set_title("种群数量变化")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 猎人总损失
    ax = axes[1, 1]
    total_hunter_loss = [
        p + v
        for p, v in zip(history["hunter_policy_loss"], history["hunter_value_loss"])
    ]
    ax.plot(timesteps, total_hunter_loss, color="red", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("总损失")
    ax.set_title("猎人总损失")
    ax.grid(True, alpha=0.3)

    # 6. 猎物总损失
    ax = axes[1, 2]
    total_prey_loss = [
        p + v for p, v in zip(history["prey_policy_loss"], history["prey_value_loss"])
    ]
    ax.plot(timesteps, total_prey_loss, color="green", linewidth=2)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("总损失")
    ax.set_title("猎物总损失")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()


def compare_checkpoints(run_dir: str, save_path: str = None):
    """对比不同检查点的性能"""
    # 查找所有历史文件
    history_files = sorted(Path(run_dir).glob("history_step_*.json"))

    if not history_files:
        print(f"错误: 在 {run_dir} 中没有找到历史文件")
        return

    print(f"找到 {len(history_files)} 个检查点")

    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("检查点性能对比", fontsize=16, fontweight="bold")

    checkpoint_steps = []
    final_hunter_rewards = []
    final_prey_rewards = []
    final_hunter_losses = []
    final_prey_losses = []

    for history_file in history_files:
        # 从文件名提取步数
        step = int(history_file.stem.split("_")[-1])
        checkpoint_steps.append(step)

        # 加载历史
        history = load_training_history(str(history_file))

        # 获取最终值
        final_hunter_rewards.append(history["hunter_mean_reward"][-1])
        final_prey_rewards.append(history["prey_mean_reward"][-1])
        final_hunter_losses.append(
            history["hunter_policy_loss"][-1] + history["hunter_value_loss"][-1]
        )
        final_prey_losses.append(
            history["prey_policy_loss"][-1] + history["prey_value_loss"][-1]
        )

    # 1. 猎人平均奖励进展
    ax = axes[0, 0]
    ax.plot(checkpoint_steps, final_hunter_rewards, marker="o", color="red", linewidth=2, markersize=8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均奖励")
    ax.set_title("猎人奖励进展")
    ax.grid(True, alpha=0.3)

    # 2. 猎物平均奖励进展
    ax = axes[0, 1]
    ax.plot(checkpoint_steps, final_prey_rewards, marker="o", color="green", linewidth=2, markersize=8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("平均奖励")
    ax.set_title("猎物奖励进展")
    ax.grid(True, alpha=0.3)

    # 3. 猎人总损失进展
    ax = axes[1, 0]
    ax.plot(checkpoint_steps, final_hunter_losses, marker="o", color="red", linewidth=2, markersize=8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("总损失")
    ax.set_title("猎人损失进展")
    ax.grid(True, alpha=0.3)

    # 4. 猎物总损失进展
    ax = axes[1, 1]
    ax.plot(checkpoint_steps, final_prey_losses, marker="o", color="green", linewidth=2, markersize=8)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("总损失")
    ax.set_title("猎物损失进展")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"检查点对比图已保存到: {save_path}")
    else:
        plt.show()

    # 打印统计信息
    print("\n" + "=" * 70)
    print("检查点性能统计")
    print("=" * 70)
    for i, step in enumerate(checkpoint_steps):
        print(f"\n步数 {step:6d}:")
        print(f"  猎人奖励: {final_hunter_rewards[i]:7.2f}")
        print(f"  猎物奖励: {final_prey_rewards[i]:7.2f}")
        print(f"  猎人损失: {final_hunter_losses[i]:7.4f}")
        print(f"  猎物损失: {final_prey_losses[i]:7.4f}")


def analyze_run(run_dir: str, output_dir: str = None):
    """分析训练运行并生成所有可视化"""
    if output_dir is None:
        output_dir = os.path.join(run_dir, "visualizations")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("训练分析")
    print("=" * 70)
    print(f"\n训练目录: {run_dir}")
    print(f"输出目录: {output_dir}")

    # 1. 绘制完整训练曲线
    print("\n[1/2] 生成训练曲线...")
    final_history_path = os.path.join(run_dir, "training_history.json")
    if os.path.exists(final_history_path):
        history = load_training_history(final_history_path)
        curves_path = os.path.join(output_dir, "training_curves.png")
        plot_training_curves(history, curves_path)
    else:
        print("警告: 未找到 training_history.json")

    # 2. 对比检查点
    print("\n[2/2] 生成检查点对比...")
    comparison_path = os.path.join(output_dir, "checkpoint_comparison.png")
    compare_checkpoints(run_dir, comparison_path)

    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    print(f"\n所有可视化已保存到: {output_dir}")


def list_runs(model_dir: str = "trained_models"):
    """列出所有训练运行"""
    print("=" * 70)
    print("可用的训练运行")
    print("=" * 70)

    run_dirs = sorted(Path(model_dir).glob("ppo_run_*"))

    if not run_dirs:
        print(f"\n没有找到训练运行在 {model_dir}")
        return

    for i, run_dir in enumerate(run_dirs, 1):
        print(f"\n{i}. {run_dir.name}")

        # 统计检查点
        checkpoints = list(run_dir.glob("model_step_*.pt"))
        print(f"   检查点数量: {len(checkpoints)}")

        # 检查是否有训练历史
        history_path = run_dir / "training_history.json"
        if history_path.exists():
            history = load_training_history(str(history_path))
            total_steps = history["timesteps"][-1] if history["timesteps"] else 0
            print(f"   总训练步数: {total_steps}")
            print(f"   最终猎人奖励: {history['hunter_mean_reward'][-1]:.2f}")
            print(f"   最终猎物奖励: {history['prey_mean_reward'][-1]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练过程可视化工具")
    parser.add_argument(
        "--mode",
        type=str,
        default="analyze",
        choices=["analyze", "curves", "compare", "list"],
        help="模式: analyze(完整分析), curves(训练曲线), compare(检查点对比), list(列出运行)",
    )
    parser.add_argument("--run_dir", type=str, help="训练运行目录")
    parser.add_argument("--history", type=str, help="训练历史文件路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--model_dir", type=str, default="trained_models", help="模型目录")
    parser.add_argument("--show", action="store_true", help="显示图表而不是保存")

    args = parser.parse_args()

    try:
        import matplotlib

        if not args.show:
            matplotlib.use("Agg")  # 无GUI后端
    except ImportError:
        print("错误: 需要安装 matplotlib")
        print("安装命令: pip install matplotlib")
        exit(1)

    if args.mode == "list":
        list_runs(args.model_dir)

    elif args.mode == "analyze":
        if not args.run_dir:
            print("错误: analyze模式需要 --run_dir 参数")
            exit(1)
        analyze_run(args.run_dir, args.output)

    elif args.mode == "curves":
        if not args.history:
            print("错误: curves模式需要 --history 参数")
            exit(1)
        history = load_training_history(args.history)
        plot_training_curves(history, args.output if not args.show else None)

    elif args.mode == "compare":
        if not args.run_dir:
            print("错误: compare模式需要 --run_dir 参数")
            exit(1)
        compare_checkpoints(args.run_dir, args.output if not args.show else None)
