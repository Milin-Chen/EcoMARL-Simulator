"""强化学习环境模块 - 重组优化版

目录结构:
- envs/          Gym环境实现
- rewards/       奖励函数 (基础、课程学习、HPO)
- training/      训练组件 (网络、PPO)
- observations.py  观测空间
- agent_controller.py  智能体控制器
"""

# 环境
try:
    from .envs import (
        EnhancedEcoMARLEnv,
        CurriculumEcoMARLEnv,
        CurriculumEcoMARLEnvHPO,
    )
except ImportError as e:
    # 如果缺少依赖（如gymnasium），提供友好提示
    import warnings
    warnings.warn(f"无法导入环境模块: {e}. 请安装: pip install gymnasium")
    EnhancedEcoMARLEnv = None
    CurriculumEcoMARLEnv = None
    CurriculumEcoMARLEnvHPO = None

# 奖励函数
from .rewards import (
    EnhancedRewardFunction,
    EnhancedRewardFunctionV2,
    CurriculumRewardFunction,
    Stage1HunterReward,
    Stage2HunterReward,
    Stage3PreyReward,
    Stage4JointReward,
    Stage1HunterRewardHPO,
    Stage3PreyRewardHPO,
    HPORewardEnhancer,
)

# 训练组件
try:
    from .training import (
        ActorCriticNetwork,
        SharedPolicy,
        PPOTrainer,
        RolloutBuffer,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"无法导入训练模块: {e}. 请安装: pip install torch")
    ActorCriticNetwork = None
    SharedPolicy = None
    PPOTrainer = None
    RolloutBuffer = None

# 其他组件
from .observations import ObservationSpace
from .agent_controller import RLAgentController, HybridController

# 兼容性别名 - 确保旧代码无需修改
EcoMARLEnv = EnhancedEcoMARLEnv
ImprovedEcoMARLEnv = EnhancedEcoMARLEnv
CollaborativeEcoMARLEnv = EnhancedEcoMARLEnv

RewardFunction = EnhancedRewardFunction
ImprovedRewardFunction = EnhancedRewardFunction
CollaborativeRewardFunction = EnhancedRewardFunction

__all__ = [
    # === 环境 ===
    'EnhancedEcoMARLEnv',
    'CurriculumEcoMARLEnv',
    'CurriculumEcoMARLEnvHPO',

    # === 奖励函数 ===
    # 基础
    'EnhancedRewardFunction',
    'EnhancedRewardFunctionV2',
    # 课程学习
    'CurriculumRewardFunction',
    'Stage1HunterReward',
    'Stage2HunterReward',
    'Stage3PreyReward',
    'Stage4JointReward',
    # HPO增强
    'Stage1HunterRewardHPO',
    'Stage3PreyRewardHPO',
    'HPORewardEnhancer',

    # === 训练组件 ===
    'ActorCriticNetwork',
    'SharedPolicy',
    'PPOTrainer',
    'RolloutBuffer',

    # === 其他 ===
    'ObservationSpace',
    'RLAgentController',
    'HybridController',

    # === 兼容性别名 ===
    'EcoMARLEnv',
    'ImprovedEcoMARLEnv',
    'CollaborativeEcoMARLEnv',
    'RewardFunction',
    'ImprovedRewardFunction',
    'CollaborativeRewardFunction',
]
