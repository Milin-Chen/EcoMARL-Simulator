import math

# 画布与颜色
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BG_COLOR = (20, 22, 28)
GRID_COLOR = (36, 40, 52)

HUNTER_COLOR = (220, 64, 64)  # 红色
PREY_COLOR = (64, 200, 96)    # 绿色
EYE_WHITE = (240, 240, 240)
EYE_PUPIL = (0, 0, 0)
FOV_COLOR = (180, 180, 180)
RAY_COLOR = (140, 140, 200)
DEBUG_PANEL_BG = (32, 36, 48)
DEBUG_PANEL_TEXT = (220, 220, 220)
ENERGY_BAR_BG = (60, 60, 60)
ENERGY_BAR_FILL = (255, 200, 64)

# FOV / 射线
DEFAULT_FOV_DEG_HUNTER = 40.0
DEFAULT_FOV_RANGE_HUNTER = 260.0
DEFAULT_FOV_DEG_PREY = 70.0
DEFAULT_FOV_RANGE_PREY = 140.0
DEFAULT_RAY_COUNT = 24

# 实体参数
DEFAULT_RADIUS = 10.0
SMOOTH_LERP = 0.18  # 插值速率（0-1）
WIGGLE_AMPLITUDE = 0.6  # 轻微蠕动效果的振幅
WIGGLE_FREQ = 2.5  # 蠕动频率

# 眼睛/瞳孔参数（更大的默认瞳孔，纯黑）
EYE_RADIUS_SCALE = 0.42
PUPIL_SCALE_BASE = 0.30
PUPIL_MIN = 0.42
PUPIL_MAX = 1.12

# 椭圆拖影参数
STRETCH_SPEED_SCALE = 0.18   # 速度对椭圆拉伸的影响系数
STRETCH_MAX_FACTOR = 0.8     # 拉伸上限（相对半径）
TRAIL_ELLIPSES = 4           # 残影椭圆个数（包含主体时另外绘制）
TRAIL_OFFSET_SCALE = 0.55    # 每个残影的偏移比例
TRAIL_ALPHA_BASE = 95        # 第一层残影透明度基准
TRAIL_ALPHA_DECAY = 0.5      # 每层递减系数

# 文本与调试
FONT_SIZE = 16
PANEL_WIDTH = 260
PANEL_MARGIN = 12
PANEL_LINE_H = 20
PANEL_ALPHA = 150  # 调试面板透明度，避免遮挡空间

# 眼睛参数（双眼）
EYE_FORWARD_SCALE = 0.7  # 双眼沿运动方向的前移比例（相对半径）
EYE_SEP_SCALE = 0.42     # 双眼左右分离比例（相对半径）

# 其他
PI = math.pi
TAU = 2 * math.pi

# 实体总量与平滑分裂参数
MAX_ENTITIES = 120
SPAWN_GROW_RATE = 0.7     # 分裂后子体半径增长速率（每秒0-1）
SPAWN_MIN_SCALE = 0.45    # 子体初始半径比例（相对目标半径）

# 预留持久化/存活参数（前端占位）
PERSIST_ENABLED = False
PERSIST_PATH = "runtime/snapshots.json"
AUTO_SAVE_INTERVAL = 10.0  # 秒

# 繁殖冷却（按类型）
BREED_CD_PREY = 2.0
BREED_CD_HUNTER = 12.0