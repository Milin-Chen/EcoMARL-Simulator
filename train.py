"""训练脚本 - PPO多智能体强化学习训练"""

import os
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict
import multiprocessing as mp

from rl_env import EcoMARLEnv  # 现在自动使用增强版
from rl_env.networks import SharedPolicy
from rl_env.ppo_trainer import PPOTrainer, RolloutBuffer
from rl_env.observations import ObservationSpace
from config import EnvConfig, AgentConfig


def setup_cpu_optimization(num_threads: int = None):
    """
    设置CPU并行优化

    Args:
        num_threads: 线程数，None表示自动检测
    """
    if num_threads is None:
        # 自动检测CPU核心数
        num_threads = mp.cpu_count()

    print(f"\n[CPU优化设置]")
    print(f"  检测到CPU核心数: {mp.cpu_count()}")
    print(f"  设置PyTorch线程数: {num_threads}")

    # 设置PyTorch的线程数
    torch.set_num_threads(num_threads)

    # 设置intra-op并行线程数（单个操作内的并行）
    torch.set_num_interop_threads(num_threads)

    # 启用MKL优化（如果可用）
    if hasattr(torch, 'set_num_mkl_threads'):
        torch.set_num_mkl_threads(num_threads)

    # 设置环境变量
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)

    print(f"  CPU优化已启用\n")


def train_ppo(
    n_hunters: int = 6,
    n_prey: int = 18,
    total_timesteps: int = 100000,
    n_steps_per_update: int = 2048,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    n_epochs: int = 10,
    batch_size: int = 64,
    save_interval: int = 10000,
    log_interval: int = 100,
    model_dir: str = "trained_models",
    device: str = "cpu",
    use_parallel: bool = True,
    num_threads: int = None,
    use_v2_rewards: bool = True,
    reward_log_interval: int = 10,
):
    """
    PPO训练主函数

    Args:
        n_hunters: 猎人数量
        n_prey: 猎物数量
        total_timesteps: 总训练步数
        n_steps_per_update: 每次更新收集的步数
        learning_rate: 学习率
        gamma: 折扣因子
        gae_lambda: GAE参数
        clip_epsilon: PPO裁剪参数
        n_epochs: 每次更新的训练轮数
        batch_size: 批次大小
        save_interval: 保存模型的间隔
        log_interval: 日志打印间隔
        model_dir: 模型保存目录
        device: 训练设备 ("cpu" 或 "cuda")
        use_parallel: 是否使用并行加速
        num_threads: CPU线程数 (None表示自动检测)
    """
    # 设置CPU优化
    if device == "cpu":
        setup_cpu_optimization(num_threads)

    # 创建保存目录
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(model_dir, f"ppo_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print("PPO多智能体强化学习训练")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  猎人数量: {n_hunters}")
    print(f"  猎物数量: {n_prey}")
    print(f"  总训练步数: {total_timesteps}")
    print(f"  每次更新步数: {n_steps_per_update}")
    print(f"  学习率: {learning_rate}")
    print(f"  设备: {device}")
    print(f"  并行加速: {use_parallel}")
    print(f"  模型保存目录: {run_dir}")
    print()

    # 创建环境
    env_config = EnvConfig()
    agent_config = AgentConfig()
    env = EcoMARLEnv(
        env_config=env_config,
        agent_config=agent_config,
        n_hunters=n_hunters,
        n_prey=n_prey,
        use_parallel=use_parallel,
        use_v2_rewards=use_v2_rewards,  # 支持V1/V2切换
    )

    # 获取观察空间维度
    obs_space_info = env.get_observation_space()
    obs_dim = obs_space_info["observation_dim"]

    print(f"观察空间维度: {obs_dim}")
    print(f"  自身状态: {obs_space_info['self_state_dim']}")
    print(f"  射线信息: {obs_space_info['ray_dim']}")
    print(f"  相对位置: {obs_space_info['relative_position_dim']}")
    print()

    # 创建策略网络
    policy = SharedPolicy(
        obs_dim=obs_dim,
        action_dim=2,
        hidden_dims=(128, 128, 64),
        device=device,
    )

    # 创建PPO训练器
    trainer = PPOTrainer(
        policy=policy,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # 训练统计
    episode_rewards = {"hunter": [], "prey": []}
    episode_lengths = []
    current_episode_rewards = {}

    # 训练历史记录（用于可视化）
    training_history = {
        "timesteps": [],
        "hunter_mean_reward": [],
        "prey_mean_reward": [],
        "hunter_policy_loss": [],
        "hunter_value_loss": [],
        "prey_policy_loss": [],
        "prey_value_loss": [],
        "n_hunters": [],
        "n_preys": [],
    }

    # 重置环境
    observations = env.reset()
    buffer = RolloutBuffer()

    # 初始化奖励累积
    for agent_id in observations.keys():
        current_episode_rewards[agent_id] = 0.0

    print("开始训练...\n")

    # 训练循环
    timestep = 0
    episode_count = 0

    while timestep < total_timesteps:
        # 收集经验
        for _ in range(n_steps_per_update):
            # 准备观察数据（添加实体类型信息）
            obs_dict = {}
            for agent_id, obs in observations.items():
                # 从环境中获取实体类型
                entity = next(
                    (e for e in env.world.entities if e.id == agent_id), None
                )
                if entity:
                    obs_dict[agent_id] = (obs, entity.type)

            # 获取动作
            actions_dict, log_probs_dict, values_dict = policy.get_actions_batch(
                obs_dict, deterministic=False
            )

            # 执行动作
            next_observations, rewards, dones, info = env.step(actions_dict)

            # 实时输出奖励
            if reward_log_interval > 0 and timestep % reward_log_interval == 0:
                hunter_rewards = [r for aid, r in rewards.items() if aid.startswith('h_')]
                prey_rewards = [r for aid, r in rewards.items() if aid.startswith('p_')]

                if hunter_rewards or prey_rewards:
                    h_mean = np.mean(hunter_rewards) if hunter_rewards else 0.0
                    p_mean = np.mean(prey_rewards) if prey_rewards else 0.0
                    h_sum = sum(hunter_rewards) if hunter_rewards else 0.0
                    p_sum = sum(prey_rewards) if prey_rewards else 0.0

                    print(f"[步{timestep:6d}] "
                          f"猎人: {h_mean:7.2f} (总:{h_sum:7.1f}, n={len(hunter_rewards)}) | "
                          f"猎物: {p_mean:7.2f} (总:{p_sum:7.1f}, n={len(prey_rewards)})")

            # 存储经验
            for agent_id in observations.keys():
                if agent_id in actions_dict:
                    entity = next(
                        (e for e in env.world.entities if e.id == agent_id), None
                    )
                    if entity:
                        buffer.add(
                            obs=observations[agent_id],
                            action=actions_dict[agent_id],
                            reward=rewards.get(agent_id, 0.0),
                            value=values_dict[agent_id],
                            log_prob=log_probs_dict[agent_id],
                            done=dones.get(agent_id, False),
                            entity_type=entity.type,
                        )

                        # 累积奖励
                        if agent_id in current_episode_rewards:
                            current_episode_rewards[agent_id] += rewards.get(
                                agent_id, 0.0
                            )

            # 处理终止的智能体
            for agent_id, done in dones.items():
                if done and agent_id in current_episode_rewards:
                    entity = next(
                        (e for e in env.world.entities if e.id == agent_id), None
                    )
                    if entity:
                        episode_rewards[entity.type].append(
                            current_episode_rewards[agent_id]
                        )
                    del current_episode_rewards[agent_id]

            # 更新观察
            observations = next_observations
            timestep += 1

            # 检查是否需要重置环境
            if len(observations) == 0 or timestep % 1000 == 0:
                # 记录episode
                episode_count += 1
                episode_lengths.append(timestep)

                # 重置环境
                observations = env.reset()
                current_episode_rewards = {
                    agent_id: 0.0 for agent_id in observations.keys()
                }

        # 计算最后状态的价值
        last_values = {}
        hunter_obs = []
        prey_obs = []

        for agent_id, obs in observations.items():
            entity = next((e for e in env.world.entities if e.id == agent_id), None)
            if entity:
                if entity.type == "hunter":
                    hunter_obs.append(obs)
                else:
                    prey_obs.append(obs)

        if hunter_obs:
            hunter_obs_tensor = torch.FloatTensor(np.array(hunter_obs)).to(
                policy.device
            )
            with torch.no_grad():
                last_values["hunter"] = (
                    policy.hunter_net.get_value(hunter_obs_tensor).mean().item()
                )

        if prey_obs:
            prey_obs_tensor = torch.FloatTensor(np.array(prey_obs)).to(policy.device)
            with torch.no_grad():
                last_values["prey"] = (
                    policy.prey_net.get_value(prey_obs_tensor).mean().item()
                )

        # 执行PPO更新
        train_stats = trainer.train_step(buffer, last_values)

        # 清空缓冲区
        buffer.reset()

        # 记录当前存活智能体的episode奖励 (用于打印统计)
        for agent_id, reward_sum in current_episode_rewards.items():
            entity = next((e for e in env.world.entities if e.id == agent_id), None)
            if entity and reward_sum != 0.0:  # 只记录非零奖励
                episode_rewards[entity.type].append(reward_sum)

        # 打印日志并记录训练历史
        if timestep % log_interval == 0 or timestep == total_timesteps:
            hunter_rewards = (
                episode_rewards["hunter"][-10:]
                if episode_rewards["hunter"]
                else [0.0]
            )
            prey_rewards = (
                episode_rewards["prey"][-10:] if episode_rewards["prey"] else [0.0]
            )

            hunter_mean_reward = np.mean(hunter_rewards)
            prey_mean_reward = np.mean(prey_rewards)

            print(f"\n[步数: {timestep}/{total_timesteps}]")
            print(f"  Episode: {episode_count}")
            print(f"  猎人平均奖励: {hunter_mean_reward:.2f} (最近10个episode)")
            print(f"  猎物平均奖励: {prey_mean_reward:.2f} (最近10个episode)")

            # 记录训练历史
            training_history["timesteps"].append(timestep)
            training_history["hunter_mean_reward"].append(hunter_mean_reward)
            training_history["prey_mean_reward"].append(prey_mean_reward)

            stats = env.simulator.get_stats()
            training_history["n_hunters"].append(stats.get("hunters", 0))
            training_history["n_preys"].append(stats.get("preys", 0))

            if train_stats:
                print(f"  训练损失:")
                for key, value in train_stats.items():
                    print(f"    {key}: {value:.4f}")

                # 记录损失
                training_history["hunter_policy_loss"].append(
                    train_stats.get("hunter_policy_loss", 0.0)
                )
                training_history["hunter_value_loss"].append(
                    train_stats.get("hunter_value_loss", 0.0)
                )
                training_history["prey_policy_loss"].append(
                    train_stats.get("prey_policy_loss", 0.0)
                )
                training_history["prey_value_loss"].append(
                    train_stats.get("prey_value_loss", 0.0)
                )

        # 保存模型和训练历史
        if timestep % save_interval == 0 or timestep == total_timesteps:
            model_path = os.path.join(run_dir, f"model_step_{timestep}.pt")
            policy.save(model_path)

            # 保存训练历史
            import json
            history_path = os.path.join(run_dir, f"history_step_{timestep}.json")
            with open(history_path, "w") as f:
                json.dump(training_history, f, indent=2)

            print(f"\n[检查点已保存: 步数={timestep}]")
            print(f"  模型: {model_path}")
            print(f"  历史: {history_path}")

    # 训练完成
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)

    # 保存最终模型和训练历史
    final_model_path = os.path.join(run_dir, "final_model.pt")
    policy.save(final_model_path)

    import json
    final_history_path = os.path.join(run_dir, "training_history.json")
    with open(final_history_path, "w") as f:
        json.dump(training_history, f, indent=2)

    print(f"\n最终模型已保存: {final_model_path}")
    print(f"训练历史已保存: {final_history_path}")

    # 清理
    env.close()

    return policy, run_dir


def evaluate_model(
    model_path: str,
    n_hunters: int = 6,
    n_prey: int = 18,
    n_episodes: int = 10,
    device: str = "cpu",
):
    """
    评估训练好的模型

    Args:
        model_path: 模型路径
        n_hunters: 猎人数量
        n_prey: 猎物数量
        n_episodes: 评估的episode数量
        device: 设备
    """
    print("=" * 70)
    print("模型评估")
    print("=" * 70)

    # 创建环境
    env_config = EnvConfig()
    agent_config = AgentConfig()
    env = EcoMARLEnv(
        env_config=env_config,
        agent_config=agent_config,
        n_hunters=n_hunters,
        n_prey=n_prey,
        use_parallel=True,
    )

    # 获取观察空间维度
    obs_space_info = env.get_observation_space()
    obs_dim = obs_space_info["observation_dim"]

    # 创建并加载策略
    policy = SharedPolicy(obs_dim=obs_dim, action_dim=2, device=device)
    policy.load(model_path)

    # 评估
    episode_rewards = {"hunter": [], "prey": []}

    for episode in range(n_episodes):
        observations = env.reset()
        episode_reward = {agent_id: 0.0 for agent_id in observations.keys()}
        done = False
        step = 0

        while not done and step < 1000:
            # 准备观察
            obs_dict = {}
            for agent_id, obs in observations.items():
                entity = next(
                    (e for e in env.world.entities if e.id == agent_id), None
                )
                if entity:
                    obs_dict[agent_id] = (obs, entity.type)

            # 获取动作（确定性）
            actions_dict, _, _ = policy.get_actions_batch(obs_dict, deterministic=True)

            # 执行动作
            observations, rewards, dones, info = env.step(actions_dict)

            # 累积奖励
            for agent_id, reward in rewards.items():
                if agent_id in episode_reward:
                    episode_reward[agent_id] += reward

            step += 1

            # 检查是否所有智能体都结束
            if len(observations) == 0:
                done = True

        # 记录奖励
        for agent_id, reward in episode_reward.items():
            entity = next(
                (e for e in env.world.entities if e.id == agent_id), None
            )
            if entity:
                episode_rewards[entity.type].append(reward)

        print(
            f"Episode {episode + 1}/{n_episodes}: "
            f"猎人奖励={np.mean([r for e, r in episode_reward.items() if 'H' in e]):.2f}, "
            f"猎物奖励={np.mean([r for e, r in episode_reward.items() if 'P' in e]):.2f}"
        )

    # 打印统计
    print("\n评估结果:")
    print(f"  猎人平均奖励: {np.mean(episode_rewards['hunter']):.2f}")
    print(f"  猎物平均奖励: {np.mean(episode_rewards['prey']):.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO多智能体强化学习训练")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--n_hunters", type=int, default=6, help="猎人数量")
    parser.add_argument("--n_prey", type=int, default=18, help="猎物数量")
    parser.add_argument(
        "--total_timesteps", type=int, default=100000, help="总训练步数"
    )
    parser.add_argument(
        "--n_steps_per_update", type=int, default=2048, help="每次更新收集的步数"
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="学习率")
    parser.add_argument("--save_interval", type=int, default=10000, help="模型保存间隔")
    parser.add_argument("--model_dir", type=str, default="trained_models", help="模型目录")
    parser.add_argument("--model_path", type=str, default=None, help="评估模型路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument("--no_parallel", action="store_true", help="禁用并行加速")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="CPU训练线程数 (None表示自动检测CPU核心数)",
    )
    parser.add_argument(
        "--use_v2_rewards",
        action="store_true",
        default=True,
        help="使用V2奖励函数 (意图性检测版，默认启用)",
    )
    parser.add_argument(
        "--use_v1_rewards",
        action="store_true",
        help="使用V1奖励函数 (原始增强版)",
    )
    parser.add_argument(
        "--reward_log_interval",
        type=int,
        default=10,
        help="实时奖励输出间隔 (每N步输出一次，0表示禁用)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        # 确定使用哪个版本的奖励函数
        use_v2 = not args.use_v1_rewards  # 默认V2，除非明确指定V1

        train_ppo(
            n_hunters=args.n_hunters,
            n_prey=args.n_prey,
            total_timesteps=args.total_timesteps,
            n_steps_per_update=args.n_steps_per_update,
            learning_rate=args.learning_rate,
            save_interval=args.save_interval,
            model_dir=args.model_dir,
            device=args.device,
            use_parallel=not args.no_parallel,
            num_threads=args.num_threads,
            use_v2_rewards=use_v2,
            reward_log_interval=args.reward_log_interval,
        )
    elif args.mode == "eval":
        if args.model_path is None:
            print("错误: 评估模式需要提供 --model_path 参数")
        else:
            evaluate_model(
                model_path=args.model_path,
                n_hunters=args.n_hunters,
                n_prey=args.n_prey,
                device=args.device,
            )
