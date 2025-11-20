"""智能体配置"""

from dataclasses import dataclass
from typing import List


@dataclass
class AgentConfig:
    """智能体配置"""

    # 猎人配置
    HUNTER_FOV_DEG: float = 70.0
    HUNTER_FOV_RANGE: float = 220.0
    HUNTER_SPEED_MIN: float = 20.0
    HUNTER_SPEED_MAX: float = 50.0  # 修改: 60.0 → 50.0 (与训练配置一致)
    HUNTER_ANGULAR_VELOCITY_MAX: float = 0.15  # 修改: 0.8 → 0.15 (与训练配置一致)

    # 猎物配置
    PREY_FOV_DEG: float = 270.0
    PREY_FOV_RANGE: float = 260.0
    PREY_SPEED_MIN: float = 15.0
    PREY_SPEED_MAX: float = 45.0  # 修改: 40.0 → 45.0 (与训练配置一致)
    PREY_ANGULAR_VELOCITY_MAX: float = 0.18  # 修改: 0.8 → 0.18 (与训练配置一致)

    # 传感器配置
    DEFAULT_RAY_COUNT: int = 24
    USE_ENTITY_FOV: bool = True

    # 动作空间配置
    SPEED_DELTA_MAX: float = 10.0  # 速度变化最大值
    ANGULAR_DELTA_MAX: float = 0.2  # 角速度变化最大值（用于动作空间定义）

    # 神经网络配置
    NN_LAYERS: List[int] = None
    NN_HAS_BIAS: bool = True

    def __post_init__(self):
        if self.NN_LAYERS is None:
            self.NN_LAYERS = [8, 32, 2]
