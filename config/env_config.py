"""环境配置"""

import math
from dataclasses import dataclass


@dataclass
class EnvConfig:
    """环境基础配置"""

    # 世界尺寸
    WINDOW_WIDTH: int = 1200
    WINDOW_HEIGHT: int = 800

    # 实体限制
    MAX_ENTITIES: int = 240
    DEFAULT_RADIUS: float = 10.0

    # 时间步长
    DT: float = 1.0 / 60.0  # 60 FPS

    # 能量系统配置
    ENERGY_BASE_METABOLISM_HUNTER: float = 0.8
    ENERGY_BASE_METABOLISM_PREY: float = 0.15
    ENERGY_PRODUCTION_PREY: float = 1.2
    ENERGY_MAX_PREY: float = 120.0
    ENERGY_MAX_HUNTER: float = 160.0
    ENERGY_SPEED_REFERENCE: float = 40.0
    ENERGY_MOVEMENT_COST: float = 0.05
    ENERGY_TURNING_COST: float = 0.08
    ENERGY_PRODUCTION_EFFICIENCY: float = 1.0

    # 捕食系统配置
    ENERGY_PREDATION_GAIN: float = 50.0
    DIGESTION_DURATION: float = 3.5
    PREDATION_BASE_RATIO: float = 0.05
    PREDATION_SIZE_BONUS: float = 0.15
    PREDATION_SPEED_BONUS: float = 0.05
    PREDATION_MIN_RADIUS: float = 30.0
    PREDATION_MAX_RADIUS: float = 100.0

    # 繁殖系统配置
    ENERGY_SPLIT_PREY: float = 120.0  # 猎物繁殖能量阈值
    ENERGY_SPLIT_HUNTER: float = 150.0  # 猎人繁殖能量阈值
    BREED_CD_PREY: float = 2.0
    BREED_CD_HUNTER: float = 12.0
    SPAWN_GROW_RATE: float = 0.7
    SPAWN_MIN_SCALE: float = 0.45

    # 数学常量
    PI: float = math.pi
    TAU: float = 2 * math.pi
