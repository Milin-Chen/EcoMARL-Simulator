"""
分阶段课程学习训练脚本 (统一版本)
Curriculum Learning Training Script (Unified Version)

功能:
- 默认使用标准课程学习 (CurriculumEcoMARLEnv)
- 可选启用HPO增强 (--enable_hpo)
- 统一使用 config/training_config.py 中的参数
- 简化代码，移除硬编码参数

4个阶段:
1. Stage 1: 猎人 vs 静止猎物
2. Stage 2: 猎人 vs 脚本猎物
3. Stage 3: 冻结猎人, 训练猎物
4. Stage 4: 联合微调
"""

# CRITICAL: These must be set before ANY other imports
import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import multiprocessing
import sys

# Force fork method instead of spawn to avoid numpy issues on macOS
# This MUST be done before importing any packages that use multiprocessing
if 'torch' not in sys.modules and 'numpy' not in sys.modules:
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.agent_config import AgentConfig
from config.env_config import EnvConfig
from config.training_config import TrainingConfig
from rl_env import (
    EnhancedEcoMARLEnv,
    CurriculumEcoMARLEnv,
    CurriculumEcoMARLEnvHPO,
    create_logger_callback,
)


# ===== 环境创建 =====


def make_vec_env(
    stage: str,
    model_dir: str,
    n_envs: int = 4,
    use_subproc: bool = True,
    enable_hpo: bool = False,
):
    """
    创建并行向量化环境

    Args:
        stage: 训练阶段 ("stage1", "stage2", "stage3", "stage4")
        model_dir: 模型目录
        n_envs: 并行环境数量
        use_subproc: 是否使用多进程
        enable_hpo: 是否启用HPO增强
    """
    # 获取阶段配置
    stage_config = TrainingConfig.get_stage_config(stage)
    total_steps = stage_config.total_timesteps

    def make_env():
        # 直接使用默认配置 (无需硬编码)
        agent_config = AgentConfig()
        env_config = EnvConfig()

        # 创建基础环境
        base_env = EnhancedEcoMARLEnv(
            agent_config=agent_config,
            env_config=env_config,
            n_hunters=stage_config.n_hunters,
            n_prey=stage_config.n_prey,
            max_steps=1000,
            use_v2_rewards=True,  # 默认使用V2奖励
        )

        # 创建课程学习环境 (选择HPO或标准版本)
        if enable_hpo:
            env = CurriculumEcoMARLEnvHPO(
                base_env=base_env,
                stage=stage,
                enable_hpo=True,
                total_steps=total_steps,
            )
        else:
            env = CurriculumEcoMARLEnv(
                base_env=base_env,
                stage=stage,
            )

        return env

    # 创建n个环境
    env_fns = [make_env for _ in range(n_envs)]

    if use_subproc and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"✓ 创建 {n_envs} 个并行环境 (SubprocVecEnv - 多进程)")
    else:
        vec_env = DummyVecEnv(env_fns)
        env_type = "单环境" if n_envs == 1 else f"{n_envs}个环境 (单进程)"
        print(f"✓ 创建 {env_type} (DummyVecEnv)")

    return vec_env


# ===== 模型加载/保存 =====


def get_model_path(stage: str, agent_type: str, model_dir: str):
    """获取模型路径"""
    return os.path.join(model_dir, f"{stage}_{agent_type}_final.zip")


def load_previous_model(stage: str, agent_type: str, env, device: str, model_dir: str):
    """加载上一阶段的模型"""
    stage_config = TrainingConfig.get_stage_config(stage)

    # 确定前置模型阶段
    if agent_type == "hunter":
        prev_stage = stage_config.load_hunter_model
    else:  # prey
        prev_stage = stage_config.load_prey_model

    if prev_stage is None:
        return None

    model_path = get_model_path(prev_stage, agent_type, model_dir)
    if not os.path.exists(model_path):
        print(f"  ⚠️  找不到 {prev_stage} 的 {agent_type} 模型: {model_path}")
        print(f"  将从头开始训练 {agent_type}")
        return None

    print(f"  ✓ 加载 {prev_stage} 的 {agent_type} 模型: {model_path}")
    model = PPO.load(model_path, env=env, device=device)
    return model


# ===== 训练函数 =====


def train_stage(
    stage: str,
    model_dir: str = "curriculum_models",
    device: str = "auto",
    reward_log_interval: int = 10,
    n_envs: int = 4,
    use_subproc: bool = True,
    enable_hpo: bool = False,
):
    """
    训练指定阶段

    Args:
        stage: 训练阶段 ("stage1", "stage2", "stage3", "stage4")
        model_dir: 模型保存目录
        device: 训练设备 ("auto", "cpu", "cuda")
        reward_log_interval: 奖励日志间隔
        n_envs: 并行环境数量
        use_subproc: 是否使用多进程
        enable_hpo: 是否启用HPO增强
    """
    # 获取阶段配置
    stage_config = TrainingConfig.get_stage_config(stage)
    train_config = TrainingConfig()

    print("\n" + "=" * 80)
    print(f"{stage_config.name}")
    print(f"描述: {stage_config.description}")
    print(f"成功标准: {stage_config.success_criteria}")
    if enable_hpo:
        print("✨ HPO增强: 启用")
    else:
        print("⚙️  模式: 标准课程学习")
    print(f"训练步数: {stage_config.total_timesteps:,}")
    print(f"学习率: {stage_config.learning_rate}")
    print(f"并行环境数: {n_envs}")
    print("=" * 80)

    # 创建模型保存目录
    os.makedirs(model_dir, exist_ok=True)

    # 创建环境
    print("\n创建环境...")
    vec_env = make_vec_env(
        stage=stage,
        model_dir=model_dir,
        n_envs=n_envs,
        use_subproc=use_subproc,
        enable_hpo=enable_hpo,
    )

    # 训练猎人
    if stage_config.train_hunters:
        print("\n训练猎人...")

        # 尝试加载前一阶段的模型
        hunter_model = load_previous_model(stage, "hunter", vec_env, device, model_dir)

        if hunter_model is None:
            # 从头创建模型
            print("  ✓ 创建新的PPO模型")
            hunter_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=stage_config.learning_rate,
                n_steps=train_config.PPO_N_STEPS,
                batch_size=train_config.PPO_BATCH_SIZE,
                n_epochs=train_config.PPO_N_EPOCHS,
                gamma=train_config.PPO_GAMMA,
                gae_lambda=train_config.PPO_GAE_LAMBDA,
                clip_range=train_config.PPO_CLIP_RANGE,
                ent_coef=train_config.PPO_ENT_COEF,
                verbose=1,
                device=device,
            )
        else:
            # 更新学习率
            hunter_model.learning_rate = stage_config.learning_rate

        # 创建LoggerCallback
        callback = create_logger_callback(
            stage=f"{stage_config.name} [猎人训练]",
            total_steps=stage_config.total_timesteps,
            update_interval=100
        )

        # 训练
        print(f"开始训练猎人 ({stage_config.total_timesteps} 步)...")
        hunter_model.learn(
            total_timesteps=stage_config.total_timesteps,
            callback=callback,
            log_interval=reward_log_interval,
            progress_bar=False,
        )

        # 保存
        hunter_path = get_model_path(stage, "hunter", model_dir)
        hunter_model.save(hunter_path)
        print(f"✅ 猎人模型已保存: {hunter_path}")

    # 训练猎物
    if stage_config.train_prey:
        print("\n训练猎物...")

        # 尝试加载前一阶段的模型
        prey_model = load_previous_model(stage, "prey", vec_env, device, model_dir)

        if prey_model is None:
            print("  ✓ 创建新的PPO模型")
            prey_model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=stage_config.learning_rate,
                n_steps=train_config.PPO_N_STEPS,
                batch_size=train_config.PPO_BATCH_SIZE,
                n_epochs=train_config.PPO_N_EPOCHS,
                gamma=train_config.PPO_GAMMA,
                gae_lambda=train_config.PPO_GAE_LAMBDA,
                clip_range=train_config.PPO_CLIP_RANGE,
                ent_coef=train_config.PPO_ENT_COEF,
                verbose=1,
                device=device,
            )
        else:
            prey_model.learning_rate = stage_config.learning_rate

        # 创建LoggerCallback
        callback = create_logger_callback(
            stage=f"{stage_config.name} [猎物训练]",
            total_steps=stage_config.total_timesteps,
            update_interval=100
        )

        print(f"开始训练猎物 ({stage_config.total_timesteps} 步)...")
        prey_model.learn(
            total_timesteps=stage_config.total_timesteps,
            callback=callback,
            log_interval=reward_log_interval,
            progress_bar=False,
        )

        prey_path = get_model_path(stage, "prey", model_dir)
        prey_model.save(prey_path)
        print(f"✅ 猎物模型已保存: {prey_path}")

    vec_env.close()
    print(f"\n✅ {stage.upper()} 训练完成!")
    print(f"成功标准: {stage_config.success_criteria}")
    print("请评估模型是否达到标准，再进入下一阶段。\n")


# ===== 主函数 =====


def main():
    parser = argparse.ArgumentParser(
        description="课程学习训练脚本 (统一版本)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练单个阶段 (标准模式)
  python train_curriculum.py --stage stage1

  # 训练单个阶段 (启用HPO增强)
  python train_curriculum.py --stage stage1 --enable_hpo

  # 训练所有阶段
  python train_curriculum.py --stages stage1 stage2 stage3 stage4

  # 高性能训练 (HPO + 多进程)
  python train_curriculum.py --stages stage1 stage2 stage3 stage4 \\
      --enable_hpo --n_envs 8 --device cpu

  # 对比实验
  python train_curriculum.py --stage stage1                  # 基线
  python train_curriculum.py --stage stage1 --enable_hpo    # HPO版本
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "stage2", "stage3", "stage4"],
        help="训练单个阶段",
    )

    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=["stage1", "stage2", "stage3", "stage4"],
        help="训练多个阶段 (按顺序)",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="curriculum_models",
        help="模型保存目录 (默认: curriculum_models)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="设备 (默认: auto)",
    )

    parser.add_argument(
        "--n_envs",
        type=int,
        default=4,
        help="并行环境数量 (默认: 4)",
    )

    parser.add_argument(
        "--no_subproc",
        action="store_true",
        help="禁用多进程 (使用DummyVecEnv)",
    )

    parser.add_argument(
        "--enable_hpo",
        action="store_true",
        help="启用HPO增强 (自适应权重 + 对抗平衡 + 距离追踪)",
    )

    parser.add_argument(
        "--reward_log_interval",
        type=int,
        default=10,
        help="奖励日志间隔 (默认: 10)",
    )

    args = parser.parse_args()

    # 确定训练阶段
    stages_to_train = []
    if args.stage:
        stages_to_train = [args.stage]
    elif args.stages:
        stages_to_train = args.stages
    else:
        # 默认：训练所有阶段
        stages_to_train = ["stage1", "stage2", "stage3", "stage4"]

    # 训练循环
    print("\n" + "=" * 80)
    print("课程学习训练")
    if args.enable_hpo:
        print("✨ HPO增强模式")
    else:
        print("⚙️  标准模式")
    print(f"训练阶段: {', '.join(stages_to_train)}")
    print("=" * 80)

    start_time = datetime.now()

    for stage in stages_to_train:
        try:
            train_stage(
                stage=stage,
                model_dir=args.model_dir,
                device=args.device,
                reward_log_interval=args.reward_log_interval,
                n_envs=args.n_envs,
                use_subproc=not args.no_subproc,
                enable_hpo=args.enable_hpo,
            )
        except KeyboardInterrupt:
            print(f"\n训练在 {stage} 阶段被中断")
            break

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"总耗时: {elapsed / 60:.1f} 分钟")
    print(f"模型保存在: {args.model_dir}/")
    print("\n运行可视化演示:")
    print(f"  python main.py")
    print(f"\n运行模型演示:")
    print(f"  python demo_curriculum_models.py")
    print()


if __name__ == "__main__":
    main()
