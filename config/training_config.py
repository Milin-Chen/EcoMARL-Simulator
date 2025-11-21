"""训练超参数配置
Training Hyperparameters Configuration

统一管理所有训练相关的超参数，避免在代码中硬编码
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class CurriculumStageConfig:
    """单个课程阶段的配置"""

    name: str
    description: str
    n_hunters: int
    n_prey: int
    total_timesteps: int
    learning_rate: float
    prey_behavior: str  # "stationary" | "flee" | "learned"
    train_hunters: bool
    train_prey: bool
    load_hunter_model: str = None  # 前置模型阶段 (如 "stage1")
    load_prey_model: str = None
    success_criteria: str = ""


@dataclass
class TrainingConfig:
    """训练超参数全局配置"""

    # ===== PPO算法参数 =====
    PPO_N_STEPS: int = 2048  # 每次更新收集的步数
    PPO_BATCH_SIZE: int = 64  # 批次大小
    PPO_N_EPOCHS: int = 10  # 每次更新的训练轮数
    PPO_GAMMA: float = 0.99  # 折扣因子
    PPO_GAE_LAMBDA: float = 0.95  # GAE参数
    PPO_CLIP_RANGE: float = 0.2  # PPO裁剪参数
    PPO_ENT_COEF: float = 0.01  # 熵系数

    # ===== 训练控制参数 =====
    DEFAULT_DEVICE: str = "auto"  # 默认设备 ("auto", "cpu", "cuda")
    DEFAULT_N_ENVS: int = 4  # 默认并行环境数
    DEFAULT_MODEL_DIR: str = "curriculum_models"  # 默认模型保存目录
    SAVE_INTERVAL: int = 10000  # 模型保存间隔
    LOG_INTERVAL: int = 100  # 日志打印间隔
    REWARD_LOG_INTERVAL: int = 10  # 实时奖励输出间隔

    # ===== HPO增强参数 =====
    HPO_ENABLE_ADAPTIVE: bool = True  # 自适应权重缩放
    HPO_ENABLE_BALANCING: bool = True  # 对抗性平衡
    HPO_ENABLE_DISTANCE: bool = True  # 距离进度追踪

    # ===== 课程学习阶段配置 =====
    @staticmethod
    def get_stage_configs() -> Dict[str, CurriculumStageConfig]:
        """获取所有阶段的配置"""
        return {
            "stage1": CurriculumStageConfig(
                name="阶段1: 猎人 vs 静止猎物",
                description="学习基础捕食行为",
                n_hunters=3,
                n_prey=6,
                total_timesteps=50000,
                learning_rate=3e-4,
                prey_behavior="stationary",
                train_hunters=True,
                train_prey=False,
                load_hunter_model=None,
                load_prey_model=None,
                success_criteria="捕食率 >80%",
            ),
            "stage2": CurriculumStageConfig(
                name="阶段2: 猎人 vs 脚本猎物",
                description="学习协作追击",
                n_hunters=3,
                n_prey=6,
                total_timesteps=100000,
                learning_rate=3e-4,
                prey_behavior="flee",
                train_hunters=True,
                train_prey=False,
                load_hunter_model="stage1",
                load_prey_model=None,
                success_criteria="捕食率 >50%, 协作行为",
            ),
            "stage3": CurriculumStageConfig(
                name="阶段3: 冻结猎人, 训练猎物",
                description="学习逃跑和集群",
                n_hunters=6,
                n_prey=18,
                total_timesteps=150000,
                learning_rate=3e-4,
                prey_behavior="learned",
                train_hunters=False,
                train_prey=True,
                load_hunter_model="stage2",
                load_prey_model=None,
                success_criteria="存活 >500步, 集群行为",
            ),
            "stage4": CurriculumStageConfig(
                name="阶段4: 联合微调",
                description="平衡对抗",
                n_hunters=6,
                n_prey=18,
                total_timesteps=150000,
                learning_rate=1e-4,
                prey_behavior="learned",
                train_hunters=True,
                train_prey=True,
                load_hunter_model="stage2",
                load_prey_model="stage3",
                success_criteria="捕食率 35-45%, 策略平衡",
            ),
        }

    @staticmethod
    def get_stage_config(stage: str) -> CurriculumStageConfig:
        """获取指定阶段的配置"""
        configs = TrainingConfig.get_stage_configs()
        if stage not in configs:
            raise ValueError(f"未知的训练阶段: {stage}. 可选: {list(configs.keys())}")
        return configs[stage]


# 默认训练配置实例
DEFAULT_TRAINING_CONFIG = TrainingConfig()
