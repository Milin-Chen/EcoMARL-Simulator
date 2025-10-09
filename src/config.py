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

# 传感器条颜色
SENSOR_SAME = (255, 140, 140)    # 同类命中（更暖）
SENSOR_OTHER = (160, 180, 255)   # 异类命中（更冷）
SENSOR_EMPTY = (90, 90, 100)     # 未命中/未知

# 捕食闪光效果
FEED_FLASH = (255, 120, 120)
FEED_FLASH_ALPHA = 140
FEED_ENERGY_DELTA_THRESHOLD = 0.5

# 捕食能量脉冲（平滑效果）
FEED_PULSE_GAIN = 0.8          # 能量增量对脉冲幅度的影响系数
FEED_PULSE_DECAY = 2.4         # 每秒衰减幅度
FEED_PULSE_MAX = 1.0           # 幅度上限
FEED_RING_MIN_SCALE = 1.2      # 脉冲环最小半径比例（相对实体半径）
FEED_RING_MAX_SCALE = 1.8      # 脉冲环最大半径比例
FEED_RING_THICKNESS = 4        # 脉冲环线宽

# FOV / 射线
DEFAULT_FOV_DEG_HUNTER = 40.0
DEFAULT_FOV_RANGE_HUNTER = 260.0
DEFAULT_FOV_DEG_PREY = 70.0
DEFAULT_FOV_RANGE_PREY = 140.0
DEFAULT_RAY_COUNT = 24

# 按类型区分的FOV默认值（前端控制，不依赖后端JSON）
HUNTER_FOV_DEG = 70.0
HUNTER_FOV_RANGE = 220.0
PREY_FOV_DEG = 90.0
PREY_FOV_RANGE = 260.0

# 是否采用实体内置的视角参数（来自数据源/JSON）进行渲染
# 开启时优先使用实体的 fov_deg/fov_range；缺失则打印错误并回退到默认值。
USE_ENTITY_FOV = True

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
# 吞咽动画（从“嘴巴到尾巴”的平滑带状高亮）
SWALLOW_COLOR = (255, 200, 180)
SWALLOW_ALPHA = 170
SWALLOW_SPEED = 1.6           # 进度每秒减少（1.0 -> 0.0），数值越大越快
SWALLOW_DECAY = 1.2           # 幅度每秒衰减
SWALLOW_LENGTH_SCALE = 2.2    # 路径长度，相对半径（越大越长）
SWALLOW_SIZE_W_SCALE = 0.9    # 吞咽带宽度（沿运动方向），相对半径
SWALLOW_SIZE_H_SCALE = 0.55   # 吞咽带高度（垂直方向），相对半径
# 速度驱动的水滴形态参数
DROPLET_MIN_SPEED = 0.35       # 低于该速度不做拉伸，保持圆形
DROPLET_SPEED_REF = 3.0        # 正常化参考速度（越大拉伸越慢）
DROPLET_MAX_LEN_SCALE = 1   # 尾部长度比例（相对半径）
DROPLET_HEAD_WIDTH_SCALE = 3 # 头部宽度比例（相对半径）
DROPLET_TAIL_WIDTH_SCALE = 0.32 # 尾部最窄半径比例（相对半径）
DROPLET_ALPHA = 160            # 水滴尾部透明度基准
DROPLET_DISCS = 7              # 水滴尾部圆盘数量（越多越平滑）
DROPLET_TRAIL_ENABLED = False  # 是否启用拖影分段效果（关闭以避免“分离感”）

# 主体透明度（软体/椭圆主体均使用）
BODY_ALPHA = 180

# 软体形变参数（果冻效果）
SOFT_BODY_NODES = 24           # 半径网节点数（越多越细腻，注意性能）
SOFT_BODY_SPRING_K = 12.0      # 弹簧强度（收敛速度）
SOFT_BODY_DAMPING = 8.0        # 阻尼（抑制震荡）
SOFT_BODY_HEAD_COMPRESS = 0.5 # 头部压缩强度（随速度放大）
SOFT_BODY_TAIL_ELONGATE = 0.18 # 尾部拉伸强度（随速度放大）
SOFT_BODY_WOBBLE_BASE = 0.2   # 基础抖动幅度（相对半径）
SOFT_BODY_WOBBLE_ANG = 0.14    # 角速度抖动增益（相对半径/|ang_vel|）