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

    # 能量系统配置 (优化: 平衡消耗与生产，防止灭绝)
    ENERGY_BASE_METABOLISM_HUNTER: float = (
        0.8  # 降低: 0.8 → 0.5 (减少基础代谢，防止猎人饿死)
    )
    ENERGY_BASE_METABOLISM_PREY: float = 0.12  # 降低: 0.15 → 0.12 (略微降低)
    ENERGY_PRODUCTION_PREY: float = (
        1.8  # 提高: 1.2 → 1.8 (增加猎物能量生产，支持更多后代)
    )
    ENERGY_MAX_PREY: float = 140.0  # 提高: 120.0 → 140.0 (更大能量储备)
    ENERGY_MAX_HUNTER: float = 1000.0  # 提高: 160.0 → 180.0 (更大能量储备)
    ENERGY_SPEED_REFERENCE: float = 40.0
    ENERGY_MOVEMENT_COST: float = 0.08  # 降低: 0.05 → 0.04 (降低移动成本)
    ENERGY_TURNING_COST: float = 0.12  # 降低: 0.08 → 0.06 (降低转向成本)
    ENERGY_PRODUCTION_EFFICIENCY: float = 1.0

    # 捕食系统配置 (优化: 提高捕食收益，降低捕食难度)
    ENERGY_PREDATION_GAIN: float = 50.0  # 提高: 50.0 → 70.0 (每次捕食获得更多能量)
    DIGESTION_DURATION: float = 2.5  # 降低: 3.5 → 2.5 (更快消化，减少饥饿时间)
    PREDATION_BASE_RATIO: float = 0.06  # 提高: 0.05 → 0.06 (稍微提高捕食率)
    PREDATION_SIZE_BONUS: float = 0.15
    PREDATION_SPEED_BONUS: float = 0.05
    PREDATION_MIN_RADIUS: float = 30.0
    PREDATION_MAX_RADIUS: float = 100.0

    # 繁殖系统配置 (优化: 降低繁殖阈值，加快种群恢复)
    ENERGY_SPLIT_PREY: float = 100.0  # 降低: 120.0 → 100.0 (更容易繁殖)
    ENERGY_SPLIT_HUNTER: float = 900.0  # 降低: 150.0 → 130.0 (更容易繁殖)
    BREED_CD_PREY: float = 1.5  # 降低: 2.0 → 1.5 (更快繁殖)
    BREED_CD_HUNTER: float = 10.0  # 降低: 12.0 → 10.0 (稍快繁殖)
    SPAWN_GROW_RATE: float = 0.7
    SPAWN_MIN_SCALE: float = 0.45

    # 数学常量
    PI: float = math.pi
    TAU: float = 2 * math.pi
