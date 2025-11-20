"""奖励函数模块"""

from .rewards_enhanced import EnhancedRewardFunction
from .rewards_enhanced_v2 import EnhancedRewardFunctionV2
from .rewards_curriculum import (
    Stage1HunterReward,
    Stage2HunterReward,
    Stage3PreyReward,
    Stage4JointReward,
    CurriculumRewardFunction,
)
from .rewards_curriculum_hpo import (
    Stage1HunterRewardHPO,
    Stage3PreyRewardHPO,
)
from .hpo_enhancements import (
    HPORewardEnhancer,
    AdaptiveRewardScaling,
    AdversarialBalancer,
    DistanceProgressTracker,
)

__all__ = [
    # 基础奖励函数
    'EnhancedRewardFunction',
    'EnhancedRewardFunctionV2',

    # 课程学习奖励
    'Stage1HunterReward',
    'Stage2HunterReward',
    'Stage3PreyReward',
    'Stage4JointReward',
    'CurriculumRewardFunction',

    # HPO增强奖励
    'Stage1HunterRewardHPO',
    'Stage3PreyRewardHPO',

    # HPO组件
    'HPORewardEnhancer',
    'AdaptiveRewardScaling',
    'AdversarialBalancer',
    'DistanceProgressTracker',
]
