"""
Stable Baselines3 Callback for Training Logger Integration
整合training_logger到SB3训练循环
"""

from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
from .training_logger import TrainingLogger, init_logger, finish_logger


class LoggerCallback(BaseCallback):
    """
    自定义Callback用于集成training_logger到SB3训练循环

    功能:
    - 在每个step调用log_step
    - 在episode结束时记录episode奖励
    - 自动在训练结束时调用finish
    """

    def __init__(self, stage: str, total_steps: int, update_interval: int = 100, verbose: int = 0):
        """
        Args:
            stage: 训练阶段名称 (如 "Stage1: 猎人 vs 静止猎物")
            total_steps: 总训练步数
            update_interval: 日志更新间隔(步数)
            verbose: SB3 verbose级别
        """
        super().__init__(verbose)
        self.stage = stage
        self.total_steps = total_steps
        self.update_interval = update_interval
        self.training_logger = None  # 改名避免与BaseCallback.logger冲突

        # Episode追踪
        self.current_episode_reward = 0.0
        self.episode_lengths = []
        self.episode_rewards = []

    def _on_training_start(self) -> None:
        """训练开始时初始化logger"""
        self.training_logger = init_logger(
            stage=self.stage,
            total_steps=self.total_steps,
            update_interval=self.update_interval
        )
        if self.verbose > 0:
            print(f"✓ TrainingLogger已初始化 (阶段: {self.stage})")

    def _on_step(self) -> bool:
        """
        每个step调用一次

        Returns:
            True to continue training
        """
        if self.training_logger is None:
            return True

        # 获取当前环境的奖励
        # self.locals包含step()返回的所有信息
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        # 累积当前episode的奖励
        if len(rewards) > 0:
            self.current_episode_reward += float(rewards[0])  # 第一个环境的奖励

        # 检查episode是否结束
        if len(dones) > 0 and dones[0]:
            # Episode结束，记录总奖励
            from .training_logger import log_step
            log_step(self.num_timesteps, self.current_episode_reward)

            # 重置episode奖励
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        else:
            # Episode未结束，正常记录step
            from .training_logger import log_step
            log_step(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        """训练结束时完成logger"""
        if self.training_logger is not None:
            finish_logger()
            if self.verbose > 0:
                print(f"\n✓ TrainingLogger已完成")


def create_logger_callback(stage: str, total_steps: int, update_interval: int = 100) -> LoggerCallback:
    """
    便捷函数: 创建LoggerCallback

    Args:
        stage: 训练阶段名称
        total_steps: 总训练步数
        update_interval: 日志更新间隔

    Returns:
        LoggerCallback实例

    使用示例:
        >>> callback = create_logger_callback("Stage1: 猎人 vs 静止猎物", 45000)
        >>> model.learn(total_timesteps=45000, callback=callback)
    """
    return LoggerCallback(stage=stage, total_steps=total_steps, update_interval=update_interval)
