"""智能体配置"""

from dataclasses import dataclass
from typing import List


@dataclass
class AgentConfig:
    """智能体配置"""

    # 猎人配置
    HUNTER_FOV_DEG: float = 70.0
    HUNTER_FOV_RANGE: float = 150.0
    HUNTER_SPEED_MIN: float = 20.0
    HUNTER_SPEED_MAX: float = 50.0  # 修改: 60.0 → 50.0 (与训练配置一致)
    HUNTER_ANGULAR_VELOCITY_MAX: float = 0.4  # 紧急修复: 0.8 → 0.4 (防止原地转圈)

    # 猎物配置 (优化: 增强威胁感知能力 + 提高转向灵敏度)
    PREY_FOV_DEG: float = 300.0  # 提高: 270.0 → 300.0 (更广视野，接近全方位感知)
    PREY_FOV_RANGE: float = 280.0  # 提高: 220.0 → 280.0 (更远探测距离，提前发现威胁)
    PREY_SPEED_MIN: float = 15.0
    PREY_SPEED_MAX: float = 60.0  # 保持: 60.0 (略快于猎人以能够逃脱)
    PREY_ANGULAR_VELOCITY_MAX: float = 0.65  # 大幅提高: 0.45 → 0.65 (提高44%, 增强紧急转向能力)

    # 传感器配置
    DEFAULT_RAY_COUNT: int = 24
    USE_ENTITY_FOV: bool = True

    # 动作空间配置
    SPEED_DELTA_MAX: float = 10.0  # 速度变化最大值
    ANGULAR_DELTA_MAX: float = 0.7  # 提高: 0.5 → 0.7 (增强转向灵活性，配合猎物角速度提升)

    # 神经网络配置
    NN_LAYERS: List[int] = None
    NN_HAS_BIAS: bool = True

    def __post_init__(self):
        if self.NN_LAYERS is None:
            self.NN_LAYERS = [8, 32, 2]
