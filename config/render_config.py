"""渲染配置"""

from dataclasses import dataclass


@dataclass
class RenderConfig:
    """前端渲染配置"""

    # 颜色配置
    BG_COLOR: tuple = (20, 22, 28)
    GRID_COLOR: tuple = (36, 40, 52)
    HUNTER_COLOR: tuple = (220, 64, 64)
    PREY_COLOR: tuple = (64, 200, 96)
    EYE_WHITE: tuple = (240, 240, 240)
    EYE_PUPIL: tuple = (0, 0, 0)
    FOV_COLOR: tuple = (180, 180, 180)
    RAY_COLOR: tuple = (140, 140, 200)
    DEBUG_PANEL_BG: tuple = (32, 36, 48)
    DEBUG_PANEL_TEXT: tuple = (220, 220, 220)
    ENERGY_BAR_BG: tuple = (60, 60, 60)
    ENERGY_BAR_FILL: tuple = (255, 200, 64)

    # 传感器颜色
    SENSOR_SAME: tuple = (255, 140, 140)
    SENSOR_OTHER: tuple = (160, 180, 255)
    SENSOR_EMPTY: tuple = (90, 90, 100)

    # 捕食效果
    FEED_FLASH: tuple = (255, 120, 120)
    FEED_FLASH_ALPHA: int = 140
    FEED_ENERGY_DELTA_THRESHOLD: float = 0.5
    FEED_PULSE_GAIN: float = 0.8
    FEED_PULSE_DECAY: float = 2.4
    FEED_PULSE_MAX: float = 1.0
    FEED_RING_MIN_SCALE: float = 1.2
    FEED_RING_MAX_SCALE: float = 1.8
    FEED_RING_THICKNESS: int = 4

    # 吞咽动画
    SWALLOW_COLOR: tuple = (255, 200, 180)
    SWALLOW_ALPHA: int = 170
    SWALLOW_SPEED: float = 1.6
    SWALLOW_DECAY: float = 1.2
    SWALLOW_LENGTH_SCALE: float = 2.2
    SWALLOW_SIZE_W_SCALE: float = 0.9
    SWALLOW_SIZE_H_SCALE: float = 0.55

    # 实体渲染
    SMOOTH_LERP: float = 0.18
    WIGGLE_AMPLITUDE: float = 0.6
    WIGGLE_FREQ: float = 2.5
    BODY_ALPHA: int = 180

    # 眼睛配置
    EYE_RADIUS_SCALE: float = 0.42
    PUPIL_SCALE_BASE: float = 0.30
    PUPIL_MIN: float = 0.42
    PUPIL_MAX: float = 1.12
    EYE_FORWARD_SCALE: float = 0.7
    EYE_SEP_SCALE: float = 0.42

    # 软体形变
    SOFT_BODY_NODES: int = 18  # 优化: 从18降至8，显著提升性能 (~10-15 FPS)
    SOFT_BODY_SPRING_K: float = 12.0
    SOFT_BODY_DAMPING: float = 8.0
    SOFT_BODY_HEAD_COMPRESS: float = 0.5
    SOFT_BODY_TAIL_ELONGATE: float = 0.18
    SOFT_BODY_WOBBLE_BASE: float = 0.2
    SOFT_BODY_WOBBLE_ANG: float = 0.14

    # UI配置
    FONT_SIZE: int = 16
    PANEL_WIDTH: int = 260
    PANEL_MARGIN: int = 12
    PANEL_LINE_H: int = 20
    PANEL_ALPHA: int = 150

    # 相机配置
    CAMERA_ZOOM_SELECTED: float = 1.8
    CAMERA_LERP: float = 0.18
