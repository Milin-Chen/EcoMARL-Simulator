"""
课程学习HPO增强Gym环境包装器
Curriculum Learning HPO-Enhanced Gym Environment Wrapper

在CurriculumEcoMARLEnv基础上集成HPO奖励增强
"""

import numpy as np
from typing import Optional
import gymnasium as gym

from .gym_env_curriculum import CurriculumEcoMARLEnv
from .gym_env_enhanced import EnhancedEcoMARLEnv
from ..rewards import CurriculumRewardFunction, Stage1HunterRewardHPO, Stage3PreyRewardHPO


class CurriculumEcoMARLEnvHPO(CurriculumEcoMARLEnv):
    """HPO增强的课程学习环境"""

    def __init__(
        self,
        base_env: EnhancedEcoMARLEnv,
        stage: str,
        enable_hpo: bool = True,
        total_steps: int = 45000,
    ):
        """
        Args:
            base_env: 基础MARL环境
            stage: "stage1" | "stage2" | "stage3" | "stage4"
            enable_hpo: 是否启用HPO增强
            total_steps: 该阶段总训练步数
        """
        # 先调用父类初始化 (会设置标准奖励函数)
        super().__init__(base_env, stage)

        self.enable_hpo = enable_hpo
        self.total_steps = total_steps

        # 如果启用HPO，替换为HPO增强的奖励函数
        if enable_hpo:
            self._replace_with_hpo_rewards()

    def _replace_with_hpo_rewards(self):
        """用HPO增强版本替换奖励函数"""
        # 获取原有的奖励函数对象
        reward_fn = self.base_env.reward_fn

        if not isinstance(reward_fn, CurriculumRewardFunction):
            print(f"⚠️  警告: 奖励函数类型不匹配，跳过HPO增强")
            return

        # 根据阶段替换特定的子奖励函数
        if self.stage == "stage1":
            # 替换猎手奖励为HPO版本
            if hasattr(reward_fn, 'hunter_reward'):
                print(f"  ✓ 启用HPO增强 - Stage1猎手奖励")
                reward_fn.hunter_reward = Stage1HunterRewardHPO(
                    total_steps=self.total_steps,
                    enable_hpo=True
                )
                self.hpo_enhancer = reward_fn.hunter_reward.hpo_enhancer

        elif self.stage == "stage2":
            # Stage2也使用猎手奖励
            if hasattr(reward_fn, 'hunter_reward'):
                print(f"  ✓ 启用HPO增强 - Stage2猎手奖励")
                reward_fn.hunter_reward = Stage1HunterRewardHPO(
                    total_steps=self.total_steps,
                    enable_hpo=True
                )
                self.hpo_enhancer = reward_fn.hunter_reward.hpo_enhancer

        elif self.stage == "stage3":
            # 替换猎物奖励为HPO版本
            if hasattr(reward_fn, 'prey_reward'):
                print(f"  ✓ 启用HPO增强 - Stage3猎物奖励")
                reward_fn.prey_reward = Stage3PreyRewardHPO(
                    total_steps=self.total_steps,
                    enable_hpo=True
                )
                self.hpo_enhancer = reward_fn.prey_reward.hpo_enhancer

        else:  # stage4
            # Stage4联合训练，替换joint_reward内的子奖励函数
            if hasattr(reward_fn, 'joint_reward'):
                joint = reward_fn.joint_reward
                print(f"  ✓ 启用HPO增强 - Stage4联合奖励")

                # 替换joint_reward内的hunter_reward和prey_reward
                if hasattr(joint, 'hunter_reward'):
                    joint.hunter_reward = Stage1HunterRewardHPO(
                        total_steps=self.total_steps,
                        enable_hpo=True
                    )
                    print(f"    - Stage4猎手奖励已替换为HPO版本")

                if hasattr(joint, 'prey_reward'):
                    joint.prey_reward = Stage3PreyRewardHPO(
                        total_steps=self.total_steps,
                        enable_hpo=True
                    )
                    print(f"    - Stage4猎物奖励已替换为HPO版本")

                # 使用猎手的增强器作为主要统计
                self.hpo_enhancer = joint.hunter_reward.hpo_enhancer if hasattr(joint, 'hunter_reward') else None
            else:
                print(f"  ⚠️  警告: Stage4没有joint_reward属性，HPO增强失败")

    def step(self, action: np.ndarray):
        """执行一步 (带HPO状态更新)"""
        # 调用父类step
        obs, reward, terminated, truncated, info = super().step(action)

        # 更新HPO状态
        if self.enable_hpo and hasattr(self, 'hpo_enhancer') and self.hpo_enhancer:
            self.hpo_enhancer.step()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """重置环境"""
        obs, info = super().reset(seed=seed, options=options)

        # 重置HPO增强器 (可选，通常在新episode开始时)
        # 注意：通常不需要在每个episode都重置，只在训练阶段改变时重置
        # if self.enable_hpo and hasattr(self, 'hpo_enhancer') and self.hpo_enhancer:
        #     self.hpo_enhancer.reset()

        return obs, info

    def get_hpo_stats(self):
        """获取HPO统计信息"""
        if self.enable_hpo and hasattr(self, 'hpo_enhancer') and self.hpo_enhancer:
            return self.hpo_enhancer.get_stats()
        return None


def create_curriculum_env_hpo(
    base_env: EnhancedEcoMARLEnv,
    stage: str,
    enable_hpo: bool = True,
    total_steps: int = 45000,
) -> gym.Env:
    """
    创建HPO增强的课程学习环境

    Args:
        base_env: 基础MARL环境
        stage: 训练阶段 ("stage1", "stage2", "stage3", "stage4")
        enable_hpo: 是否启用HPO增强
        total_steps: 该阶段总训练步数

    Returns:
        CurriculumEcoMARLEnvHPO实例
    """
    return CurriculumEcoMARLEnvHPO(
        base_env=base_env,
        stage=stage,
        enable_hpo=enable_hpo,
        total_steps=total_steps,
    )
