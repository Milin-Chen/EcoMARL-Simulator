from __future__ import annotations

import math
import time
import logging
from collections import deque
from typing import Dict, Optional, Tuple, Any
from types import SimpleNamespace

import pygame

import config
from models import WorldState, EntityState, EntityType, RayHit

# 统一日志（可由宿主程序覆盖配置）
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class RendererError(Exception):
    """前端渲染器致命错误，要求宿主程序进行兜底处理或退出。"""
    pass


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PygameRenderer:
    def __init__(self):
        try:
            pygame.init()
            pygame.display.set_caption("智能生态模拟器 - PyGame 前端")
            self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Menlo", config.FONT_SIZE)
        except Exception as exc:
            logger.exception("PyGame 初始化失败: %s", exc)
            raise RendererError("无法初始化PyGame显示窗口") from exc

        self.running = True
        self.paused = False
        # 视线仅在选中实体时显示
        self.show_rays = False
        self.show_debug = True

        self.selected_id: Optional[str] = None
        # 平滑插值缓存：id -> (x, y, angle)
        self._smooth: Dict[str, Tuple[float, float, float]] = {}
        # 运动轨迹（拖动效果）：id -> deque[(x,y)]
        self._trail: Dict[str, deque] = {}
        # 绘制前一帧位置，用于基于运动方向的拉伸
        self._prev_draw: Dict[str, Tuple[float, float]] = {}
        # 软体形变状态缓存：每个实体的节点半径与速度
        self._soft_radii: Dict[str, list[float]] = {}
        self._soft_vels: Dict[str, list[float]] = {}
        # 软体开关（保留为始终开启，移除水滴模式）
        self._soft_enabled: bool = True

        # 可调参数
        self._ray_count_delta = 0
        self._fov_range_scale = 1.0
        self._fov_mode = "wedge"  # wedge | lines | off
        self.show_sensors = True
        self._prev_energy: Dict[str, float] = {}
        self._feed_pulse: Dict[str, float] = {}
        self._swallow_prog: Dict[str, float] = {}
        self._swallow_amp: Dict[str, float] = {}
        # 事件驱动的前端状态：用于JSON事件触发的效果与计数
        self._spawn_override: Dict[str, float] = {}
        self._tick_swallow: Dict[str, bool] = {}
        self._stats: Dict[str, int] = {"predations": 0, "births": 0}
        # 每个捕食者的击杀计数
        self._predation_count: Dict[str, int] = {}

        # 性能与异常监控
        self._last_dt_sec: float = 0.0
        self._slow_frame_warn_threshold: float = 0.120  # 单帧>120ms发警告
        self._stale_world_frames: int = 0
        self._stale_world_limit: int = 120  # 连续2秒无有效world则告警

        # 外部提供的动作帧回放（用于在前端复现你提供的连续帧）
        self._ghost_frames: Optional[list[Tuple[float, float, float]]] = None  # [(x, y, angle), ...]
        self._ghost_play: bool = False
        self._ghost_idx: int = 0
        self._ghost_rate: float = 24.0  # 每秒播放帧数
        self._ghost_time_accum: float = 0.0

    # -------- 公共API（便于在其他主程序中调用） --------
    def update_frame(self, world: Optional[WorldState]) -> None:
        """
        单帧更新：事件处理 + 绘制 + 翻转 + tick。
        宿主程序可循环调用此方法驱动前端。
        """
        if world is None:
            self._stale_world_frames += 1
            if self._stale_world_frames == self._stale_world_limit:
                logger.warning("前端连续%dx未获取到有效世界状态", self._stale_world_limit)
            # 即便无world，也要处理事件与tick以保持窗口响应
        else:
            self._stale_world_frames = 0
        try:
            # 事件与绘制
            if world is not None:
                self.handle_events(world)
                if not self.paused:
                    self.draw_world(world)
            else:
                # 无world时，仅清屏与事件维持
                self.handle_events(WorldState(tick=0, entities=[]))
                self._draw_grid()
            # 翻转与计时
            try:
                pygame.display.flip()
            except pygame.error as e:
                logger.exception("显示翻转失败: %s", e)
                raise RendererError("显示翻转失败") from e
            dt_ms = self.tick()
            self._last_dt_sec = dt_ms / 1000.0
            if self._last_dt_sec > self._slow_frame_warn_threshold:
                logger.warning("帧耗时过高: %.1fms", dt_ms)
        except RendererError:
            # 上层已抛致命错误，直接重抛
            self.running = False
            raise
        except Exception as exc:
            logger.exception("前端更新帧异常: %s", exc)
            self.running = False
            raise RendererError("前端更新帧异常") from exc

    def run_loop(self, source: Any) -> None:
        """
        阻塞式运行循环：从数据源读取并驱动前端，供独立运行或宿主直接调用。
        source 需提供 poll() -> Optional[WorldState]
        """
        last_frame: Optional[WorldState] = None
        try:
            while self.running:
                world = None
                try:
                    world = source.poll()
                except Exception as exc:
                    # 数据源异常不直接崩溃前端，但要提示并保持事件响应
                    logger.exception("数据源poll异常: %s", exc)
                if world is None:
                    world = last_frame
                if world is None:
                    # 首帧未就绪，轻微等待以避免busy-loop
                    time.sleep(0.02)
                    # 仍要让窗口可响应
                    self.update_frame(None)
                    continue
                last_frame = world
                self.update_frame(world)
        finally:
            try:
                pygame.quit()
            except Exception:
                # 退出时不再上报异常
                pass

    @staticmethod
    def run_with(source: Any) -> None:
        """便捷入口：创建并运行前端循环。"""
        renderer = PygameRenderer()
        renderer.run_loop(source)

    # 上下文管理，便于宿主使用 with 保证退出
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            pygame.quit()
        finally:
            # 返回False以传播异常给宿主处理
            return False

    def _color_for(self, e: EntityState):
        return config.HUNTER_COLOR if e.type == "hunter" else config.PREY_COLOR

    def _fov_params(self, e: EntityState) -> Tuple[float, float]:
        """根据配置决定使用实体内置FOV或默认FOV；实体缺失属性时打印错误并回退。"""
        use_entity = bool(getattr(config, "USE_ENTITY_FOV", True))
        if use_entity:
            deg = e.fov_deg
            rng = e.fov_range
            if deg is None or rng is None:
                missing = []
                if deg is None:
                    missing.append("fov_deg")
                if rng is None:
                    missing.append("fov_range")
                try:
                    logger.error("实体FOV属性缺失: id=%s type=%s missing=%s，使用默认参数渲染", e.id, e.type, ",".join(missing))
                except Exception:
                    # logger 出错时不阻断渲染
                    pass
                if e.type == "hunter":
                    deg = config.HUNTER_FOV_DEG
                    rng = config.HUNTER_FOV_RANGE
                else:
                    deg = config.PREY_FOV_DEG
                    rng = config.PREY_FOV_RANGE
        else:
            if e.type == "hunter":
                deg = config.HUNTER_FOV_DEG
                rng = config.HUNTER_FOV_RANGE
            else:
                deg = config.PREY_FOV_DEG
                rng = config.PREY_FOV_RANGE
        rng = float(rng) * self._fov_range_scale
        return float(deg), float(rng)

    def _draw_sensor_strip(self, e: EntityState):
        """在屏幕右上角绘制传感器条，并在旁边绘制类别颜色说明。"""
        if not self.show_sensors or not e.rays:
            return
        # 面板参数
        margin = 10
        cell = 12
        gap = 4
        h = len(e.rays) * (cell + gap) + gap
        p_w = cell + gap * 2
        # 右上角定位
        screen_w = self.screen.get_width()
        x0 = screen_w - p_w - margin
        y0 = margin
        # 面板背景与边框
        panel = pygame.Surface((p_w, h), pygame.SRCALPHA)
        panel.fill((30, 32, 40, 120))
        self.screen.blit(panel, (x0, y0))
        pygame.draw.rect(self.screen, (180, 180, 190), pygame.Rect(x0, y0, p_w, h), 1)

        # 逐个 ray 绘制方块（纵向柱）
        for i, r in enumerate(e.rays):
            color = config.SENSOR_EMPTY
            if r.hit_type == "hunter" and e.type == "hunter":
                color = config.SENSOR_SAME
            elif r.hit_type == "prey" and e.type == "prey":
                color = config.SENSOR_SAME
            elif r.hit_type in ("hunter", "prey"):
                color = config.SENSOR_OTHER
            cy = y0 + gap + i * (cell + gap)
            rect = pygame.Rect(x0 + gap, cy, cell, cell)
            pygame.draw.rect(self.screen, color, rect)
            if r.hit_id and r.hit_id == e.target_id:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # 类别颜色说明（靠面板左侧，避免越界）
        legend_items = [
            ("same", config.SENSOR_SAME),
            ("different", config.SENSOR_OTHER),
            ("null", config.SENSOR_EMPTY),
        ]
        # 计算说明框尺寸
        lx = max(70, int(self.font.size("null")[0] + 30))
        ly = gap * 2 + len(legend_items) * (cell + gap)
        legend = pygame.Surface((lx, ly), pygame.SRCALPHA)
        legend.fill((26, 28, 36, 120))
        # 放在面板左侧
        lg_x = x0 - lx - 8
        lg_y = y0
        self.screen.blit(legend, (lg_x, lg_y))
        pygame.draw.rect(self.screen, (160, 160, 170), pygame.Rect(lg_x, lg_y, lx, ly), 1)
        # 绘制每行说明
        for i, (label, color) in enumerate(legend_items):
            row_y = lg_y + gap + i * (cell + gap)
            swatch = pygame.Rect(lg_x + gap, row_y, cell, cell)
            pygame.draw.rect(self.screen, color, swatch)
            text_img = self.font.render(label, True, (220, 220, 228))
            self.screen.blit(text_img, (lg_x + gap + cell + 6, row_y - 1))

    def _fmt_age(self, age_seconds: float) -> str:
        # 将秒数格式化为 mm:ss.ss
        m = int(max(0.0, age_seconds) // 60)
        s = max(0.0, age_seconds) % 60.0
        return f"{m:02d}:{s:05.2f}"

    def _lerp_state(self, e: EntityState):
        prev = self._smooth.get(e.id)
        wiggle = config.WIGGLE_AMPLITUDE
        if not prev:
            self._smooth[e.id] = (e.x, e.y, e.angle)
            return e.x, e.y, e.angle
        px, py, pa = prev
        nx = px + (e.x - px) * config.SMOOTH_LERP
        ny = py + (e.y - py) * config.SMOOTH_LERP
        na = pa + (e.angle - pa) * config.SMOOTH_LERP
        # 轻微的蠕动效果
        t = pygame.time.get_ticks() / 1000.0
        nx += math.sin(t * config.WIGGLE_FREQ + hash(e.id) % 10) * wiggle
        ny += math.cos(t * config.WIGGLE_FREQ + hash(e.id) % 10) * wiggle
        self._smooth[e.id] = (nx, ny, na)
        return nx, ny, na

    def _draw_grid(self):
        # 简易背景网格，便于观察运动
        # 使用 config.BG_COLOR 作为背景色
        self.screen.fill(config.BG_COLOR)
        step = 40
        for x in range(0, config.WINDOW_WIDTH, step):
            pygame.draw.line(self.screen, (30, 30, 40), (x, 0), (x, config.WINDOW_HEIGHT), 1)
        for y in range(0, config.WINDOW_HEIGHT, step):
            pygame.draw.line(self.screen, (30, 30, 40), (0, y), (config.WINDOW_WIDTH, y), 1)

    # 仅保留软体实现（已删除水滴相关函数与效果）

    def _ensure_soft_state(self, e_id: str, r: float):
        n = max(8, int(getattr(config, "SOFT_BODY_NODES", 24)))
        if e_id not in self._soft_radii or len(self._soft_radii[e_id]) != n:
            self._soft_radii[e_id] = [r for _ in range(n)]
            self._soft_vels[e_id] = [0.0 for _ in range(n)]

    def _draw_soft_body(self, e: EntityState, x: float, y: float, move_a: float, r: float, color: Tuple[int, int, int]):
        """果冻式软体主体：节点半径弹簧-阻尼积分并绘制半透明多边形。
        - 低速保持圆形半透明
        - 中高速：前压后伸，随角速度产生轻微抖动
        """
        # 软体使用自身速度参数，不依赖水滴配置
        min_s = 0.35
        ref_s = 3.0
        s = clamp(e.speed / ref_s, 0.0, 1.0)
        n = max(8, int(getattr(config, "SOFT_BODY_NODES", 24)))
        self._ensure_soft_state(e.id, r)
        radii = self._soft_radii[e.id]
        vels = self._soft_vels[e.id]

        if e.speed < min_s:
            # 重置趋近圆形，避免停住后残留拉伸
            for i in range(n):
                radii[i] = r
                vels[i] = 0.0
            d = int(r * 2)
            surf = pygame.Surface((d, d), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*color, getattr(config, "BODY_ALPHA", 180)), (int(d/2), int(d/2)), int(r))
            rect = surf.get_rect(center=(int(x), int(y)))
            self.screen.blit(surf, rect)
            return

        # 目标形变参数（避免“分离/花生形”）：改为加性形变并使用平滑权重
        major_scale = 1.0 + 0.42 * s
        minor_scale = 1.0 - 0.26 * s
        head_c = getattr(config, "SOFT_BODY_HEAD_COMPRESS", 0.14) * s
        tail_e = getattr(config, "SOFT_BODY_TAIL_ELONGATE", 0.18) * s
        wobble_base = getattr(config, "SOFT_BODY_WOBBLE_BASE", 0.08) * r
        wobble_ang = getattr(config, "SOFT_BODY_WOBBLE_ANG", 0.06) * r * min(1.0, abs(e.angular_velocity))
        # 参数二次收敛：降低形变幅度以减少分离现象
        major_scale = 1.0 + 0.22 * s
        minor_scale = 1.0 - 0.10 * s
        head_c = getattr(config, "SOFT_BODY_HEAD_COMPRESS", 0.08) * s
        tail_e = getattr(config, "SOFT_BODY_TAIL_ELONGATE", 0.12) * s
        wobble_base = getattr(config, "SOFT_BODY_WOBBLE_BASE", 0.05) * r
        wobble_ang = getattr(config, "SOFT_BODY_WOBBLE_ANG", 0.04) * r * min(1.0, abs(e.angular_velocity))
        k = float(getattr(config, "SOFT_BODY_SPRING_K", 12.0))
        dmp = float(getattr(config, "SOFT_BODY_DAMPING", 8.0))
        dt = max(1.0/60.0, self._last_dt_sec)
        t = pygame.time.get_ticks() / 1000.0

        # 更新每个节点半径
        points: list[Tuple[int, int]] = []
        for i in range(n):
            phi = move_a + (i / n) * (config.TAU)
            # 椭圆基准缩放（沿速度方向更长，横向稍窄）
            dir_w = abs(math.cos(phi - move_a))  # 0..1
            base_scale = minor_scale + (major_scale - minor_scale) * dir_w
            # 平滑权重，避免侧面过窄导致“腰分离”
            w = math.cos(phi - move_a)  # -1..1，正值：前方；负值：后方
            head_w = max(0.0, w)
            tail_w = max(0.0, -w)
            # 使用平方权重减缓侧向影响，并改为加性形变
            comp = (head_c * (head_w ** 2))
            elong = (tail_e * (tail_w ** 2))
            shape_scale = base_scale
            target = r * shape_scale - r * comp + r * elong
            # 抖动（基于时间与角速度），降低幅度避免形状散裂
            wobble = 0.6 * (wobble_base + wobble_ang) * math.sin(t * config.WIGGLE_FREQ + i * 0.7)
            target += wobble
            # 与圆形插值，防止过度前后拉伸造成分离
            target = r + (target - r) * 0.65

            # 弹簧-阻尼积分
            dv = (target - radii[i]) * k * dt - vels[i] * dmp * dt
            vels[i] += dv
            radii[i] += vels[i] * dt
            # 简单约束避免过度形变
            # 约束避免过度变窄或过长导致视觉分离
            radii[i] = clamp(radii[i], r * 0.75, r * 1.9)
            # 二次约束：进一步收紧范围以保证连续性
            radii[i] = clamp(radii[i], r * 0.88, r * 1.5)

            px = int(x + math.cos(phi) * radii[i])
            py = int(y + math.sin(phi) * radii[i])
            points.append((px, py))

        # 在透明画布上绘制多边形后 blit 到屏幕
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        w = max_x - min_x + 8
        h = max_y - min_y + 8
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        local_pts = [(p[0] - min_x + 4, p[1] - min_y + 4) for p in points]
        pygame.draw.polygon(surf, (*color, getattr(config, "BODY_ALPHA", 180)), local_pts)
        self.screen.blit(surf, (min_x - 4, min_y - 4))

    def _draw_swallow_band(self, x: float, y: float, move_a: float, r: float, prog: float, amp: float):
        """吞咽动画：一条沿运动方向移动的高亮带，从头到尾。
        prog: 进度(1->0)，1在嘴部，0在尾部
        amp: 幅度(0->1)，影响透明度
        """
        dirx, diry = math.cos(move_a), math.sin(move_a)
        head_x = x + dirx * r
        head_y = y + diry * r
        path_len = r * config.SWALLOW_LENGTH_SCALE
        # 从嘴到尾的插值位置
        cx = head_x - dirx * (path_len * (1.0 - prog))
        cy = head_y - diry * (path_len * (1.0 - prog))
        bw = max(6, int(r * config.SWALLOW_SIZE_W_SCALE))
        bh = max(4, int(r * config.SWALLOW_SIZE_H_SCALE))
        alpha = int(config.SWALLOW_ALPHA * clamp(amp, 0.0, 1.0))
        surf = pygame.Surface((bw + 6, bh + 6), pygame.SRCALPHA)
        pygame.draw.ellipse(surf, (*config.SWALLOW_COLOR, alpha), pygame.Rect(3, 3, bw, bh))
        rot = pygame.transform.rotate(surf, math.degrees(move_a))
        rect = rot.get_rect(center=(int(cx), int(cy)))
        self.screen.blit(rot, rect)

    def _process_events(self, world: WorldState) -> None:
        """处理来自JSON的事件：捕食/繁殖/成长。使效果在前端渲染层实现。"""
        # 本帧事件触发表（用于跳过能量差的隐式触发）
        self._tick_swallow.clear()
        # 后端计数（若提供）优先
        try:
            if getattr(world, "counters", None) and isinstance(world.counters, dict):
                self._stats["predations"] = int(world.counters.get("predations", self._stats.get("predations", 0)))
                self._stats["births"] = int(world.counters.get("births", self._stats.get("births", 0)))
                pk = world.counters.get("predator_kills")
                if isinstance(pk, dict):
                    # 用后端提供的每捕食者击杀数覆盖/合并
                    for k, v in pk.items():
                        try:
                            self._predation_count[str(k)] = int(v)
                        except Exception:
                            continue
        except Exception:
            pass
        # 逐事件处理
        for ev in getattr(world, "events", []) or []:
            ev_type = getattr(ev, "type", None)
            if ev_type == "predation":
                actor_id = getattr(ev, "actor_id", None)
                if actor_id:
                    gain = float(getattr(ev, "energy_gain", 1.0) or 1.0)
                    # 吞咽带与喂食脉冲
                    self._feed_pulse[actor_id] = min(config.FEED_PULSE_MAX, self._feed_pulse.get(actor_id, 0.0) + gain * config.FEED_PULSE_GAIN)
                    self._swallow_prog[actor_id] = 1.0
                    self._swallow_amp[actor_id] = max(self._swallow_amp.get(actor_id, 0.0), min(1.0, 0.6 + gain * 0.3))
                    self._tick_swallow[actor_id] = True
                    self._stats["predations"] = self._stats.get("predations", 0) + 1
                    # 按捕食者统计击杀次数
                    self._predation_count[actor_id] = self._predation_count.get(actor_id, 0) + 1
            elif ev_type == "breed":
                child = getattr(ev, "child", None)
                cid = None
                if isinstance(child, dict):
                    cid = child.get("id")
                if cid:
                    # 让新子体从小到大长成（即便后端未提供spawn_progress）
                    self._spawn_override[cid] = 0.0
                    self._stats["births"] = self._stats.get("births", 0) + 1

    def _draw_entity(self, e: EntityState, gaze_dir: Tuple[float, float], gaze_dist: float):
        x, y, a = self._lerp_state(e)
        color = self._color_for(e)
        # 平滑分裂缩放：render半径随spawn_progress从小到大
        base_r = e.radius
        # 优先使用事件驱动的成长覆盖，其次使用后端提供的spawn_progress
        sp = self._spawn_override.get(e.id, float(getattr(e, "spawn_progress", 1.0)))
        scale = config.SPAWN_MIN_SCALE + (1.0 - config.SPAWN_MIN_SCALE) * sp
        r = base_r * scale
        # 基于运动方向的拉伸角度
        px, py = self._prev_draw.get(e.id, (x, y))
        mvx, mvy = (x - px), (y - py)
        mvlen = math.hypot(mvx, mvy)
        move_a = math.atan2(mvy, mvx) if mvlen > 0.3 else a

        # 主体仅为软体，无水滴拖影

        # FOV 显示（仅选中时；模式可切换；可关闭）
        fov_deg, fov_range = self._fov_params(e)
        if self.selected_id == e.id and self._fov_mode != "off":
            if self._fov_mode == "wedge":
                half = math.radians(fov_deg) / 2
                p1 = (x, y)
                p2 = (x + math.cos(a - half) * fov_range, y + math.sin(a - half) * fov_range)
                p3 = (x + math.cos(a + half) * fov_range, y + math.sin(a + half) * fov_range)
                pygame.draw.polygon(self.screen, (60, 60, 80), [p1, p2, p3], 1)
            else:
                # 线束模式：从实体中心向FOV边缘绘制若干条线以表示视域
                half = math.radians(fov_deg) / 2
                lines = 12
                for i in range(lines + 1):
                    t = -half + (half * 2) * (i / max(1, lines))
                    ex = x + math.cos(a + t) * fov_range
                    ey = y + math.sin(a + t) * fov_range
                    pygame.draw.line(self.screen, (60, 60, 80), (x, y), (ex, ey), 1)

        # 仅绘制软体主体
        self._draw_soft_body(e, x, y, move_a, r, color)


        # 能量条（仅选中时显示）
        if self.selected_id == e.id:
            bar_w = r * 2
            bar_h = 5
            bg_rect = pygame.Rect(int(x - r), int(y - r - 10), int(bar_w), bar_h)
            fill_w = int(bar_w * clamp(e.energy / max(1.0, e.split_energy), 0.0, 1.0))
            fill_rect = pygame.Rect(int(x - r), int(y - r - 10), fill_w, bar_h)
            pygame.draw.rect(self.screen, config.ENERGY_BAR_BG, bg_rect)
            pygame.draw.rect(self.screen, config.ENERGY_BAR_FILL, fill_rect)

        # 双眼与瞳孔（看向最近实体；距离影响瞳孔大小）
        pupil = clamp(1.0 - gaze_dist / max(1.0, fov_range), 0.1, 0.9)
        eye_r = r * config.EYE_RADIUS_SCALE
        # 双眼位置：沿运动方向前移，再左右分离
        fx = math.cos(move_a) * (r * config.EYE_FORWARD_SCALE)
        fy = math.sin(move_a) * (r * config.EYE_FORWARD_SCALE)
        perp_x = -math.sin(move_a) * (r * config.EYE_SEP_SCALE)
        perp_y = math.cos(move_a) * (r * config.EYE_SEP_SCALE)
        left_eye = (x + fx - perp_x, y + fy - perp_y)
        right_eye = (x + fx + perp_x, y + fy + perp_y)
        for cx, cy in (left_eye, right_eye):
            pygame.draw.circle(self.screen, config.EYE_WHITE, (int(cx), int(cy)), int(eye_r))
            # 仅在视野(FOV)范围内偏移瞳孔方向
            gx, gy = gaze_dir
            ga = math.atan2(gy, gx)
            half = math.radians(fov_deg) / 2.0
            # 将方向角限制到 [a-half, a+half]
            diff = (ga - a + math.pi) % (2 * math.pi) - math.pi
            if diff > half:
                ga = a + half
            elif diff < -half:
                ga = a - half
            dirx, diry = math.cos(ga), math.sin(ga)
            ox = dirx * (eye_r * 0.35)
            oy = diry * (eye_r * 0.35)
            pupil_center = (int(cx + ox), int(cy + oy))
            pupil_sz = int(eye_r * max(config.PUPIL_MIN, min(config.PUPIL_MAX, config.PUPIL_SCALE_BASE * pupil)))
            pygame.draw.circle(self.screen, config.EYE_PUPIL, pupil_center, pupil_sz)

        # 射线（仅选中时显示）
        if self.selected_id == e.id and e.rays:
            for h in e.rays:
                dx = math.cos(h.angle) * h.distance
                dy = math.sin(h.angle) * h.distance
                end = (x + dx, y + dy)
                pygame.draw.line(self.screen, config.RAY_COLOR, (x, y), end, 1)

        # 捕食能量脉冲（平滑）与吞咽带：仅对捕食者生效，且对新生子体抑制误触发
        prev_e = self._prev_energy.get(e.id, e.energy)
        delta_e = e.energy - prev_e
        is_spawnling = sp < 0.98
        if e.type == "hunter":
            pulse_amp = self._feed_pulse.get(e.id, 0.0)
            swallow_p = self._swallow_prog.get(e.id, 0.0)
            swallow_amp = self._swallow_amp.get(e.id, 0.0)
            # 若本帧已有事件触发吞咽，则跳过能量差的隐式触发
            if (not self._tick_swallow.get(e.id)) and delta_e > config.FEED_ENERGY_DELTA_THRESHOLD and not is_spawnling:
                pulse_amp = min(config.FEED_PULSE_MAX, pulse_amp + delta_e * config.FEED_PULSE_GAIN)
                swallow_p = 1.0
                swallow_amp = max(swallow_amp, min(1.0, 0.5 + delta_e * 0.4))
            # 脉冲衰减
            if pulse_amp > 0.0:
                pulse_amp = max(0.0, pulse_amp - config.FEED_PULSE_DECAY * max(0.0, self._last_dt_sec))
                scale = config.FEED_RING_MIN_SCALE + (config.FEED_RING_MAX_SCALE - config.FEED_RING_MIN_SCALE) * pulse_amp
                ring_r = int(r * scale)
                alpha = int(config.FEED_FLASH_ALPHA * (0.6 + 0.4 * pulse_amp))
                ring = pygame.Surface((ring_r * 2 + 8, ring_r * 2 + 8), pygame.SRCALPHA)
                pygame.draw.circle(ring, (*config.FEED_FLASH, alpha), (ring_r + 4, ring_r + 4), ring_r, config.FEED_RING_THICKNESS)
                rect = ring.get_rect(center=(int(x), int(y)))
                self.screen.blit(ring, rect)
            self._feed_pulse[e.id] = pulse_amp
            # 吞咽动画（从嘴到尾）
            if swallow_p > 0.0 and swallow_amp > 0.02:
                self._draw_swallow_band(x, y, move_a, r, swallow_p, swallow_amp)
                swallow_p = max(0.0, swallow_p - config.SWALLOW_SPEED * max(0.0, self._last_dt_sec))
                swallow_amp = max(0.0, swallow_amp - config.SWALLOW_DECAY * max(0.0, self._last_dt_sec))
            self._swallow_prog[e.id] = swallow_p
            self._swallow_amp[e.id] = swallow_amp
        else:
            # 非捕食者不显示红圈与吞咽效果
            self._feed_pulse[e.id] = 0.0
            self._swallow_prog[e.id] = 0.0
            self._swallow_amp[e.id] = 0.0

        # 选中高亮
        if self.selected_id == e.id:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(r + 3), 1)
        # 更新前一帧位置
        self._prev_draw[e.id] = (x, y)
        self._prev_energy[e.id] = e.energy

    def _draw_debug_panel(self, world: WorldState):
        if not self.show_debug:
            return
        panel_h = config.PANEL_MARGIN * 2 + config.PANEL_LINE_H * 12
        panel = pygame.Surface((config.PANEL_WIDTH, panel_h))
        panel.set_alpha(config.PANEL_ALPHA)
        panel.fill(config.DEBUG_PANEL_BG)
        # 置于左上角，避免遮挡主要空间
        self.screen.blit(panel, (0, 0))

        def write(line: int, text: str):
            img = self.font.render(text, True, config.DEBUG_PANEL_TEXT)
            self.screen.blit(img, (config.PANEL_MARGIN, config.PANEL_MARGIN + line * config.PANEL_LINE_H))

        write(0, f"Tick: {world.tick} | Entities: {len(world.entities)}")
        write(1, f"Paused: {self.paused} | Rays: {self.show_rays} | Debug: {self.show_debug}")
        write(2, f"FOV scale: {self._fov_range_scale:.2f} | Ray delta: {self._ray_count_delta}")
        write(3, "Body: soft")

        sel = next((e for e in world.entities if e.id == self.selected_id), None)
        if sel:
            write(4, f"Selected: {sel.id} ({sel.type})")
            if sel.type == "hunter":
                write(5, f"Energy: {sel.energy:.1f} / split {sel.split_energy:.1f}")
                write(6, f"Speed: {sel.speed:.1f} | AngVel: {sel.angular_velocity:.2f}")
                write(7, f"alive_time: {self._fmt_age(sel.age)} | Gen: {sel.generation}")
                write(8, f"Children: {sel.offspring_count}")
                deg, rng = self._fov_params(sel)
                write(9, f"FOV: {deg:.1f} deg / {rng:.1f} px")
                kills = self._predation_count.get(sel.id, 0)
                write(10, f"Kills: {kills}")
                write(11, f"Digestion: {sel.digestion:.2f} | Target: {sel.target_id}")
            else:
                write(5, f"Energy: {sel.energy:.1f} / split {sel.split_energy:.1f}")
                write(6, f"Speed: {sel.speed:.1f}")
                write(7, f"alive_time: {self._fmt_age(sel.age)} | Gen: {sel.generation}")
                write(8, f"Children: {sel.offspring_count}")
                deg, rng = self._fov_params(sel)
                write(9, f"FOV: {deg:.1f} deg / {rng:.1f} px")

    def _pick_entity(self, world: WorldState, pos):
        x, y = pos
        closest = None
        best_d2 = 1e9
        for e in world.entities:
            # 仅过滤已死亡的捕食者；零能量的猎物仍可选中
            if e.type == "hunter" and e.energy <= 0:
                continue
            # 使用与绘制一致的平滑位置进行点击检测
            sx, sy, _ = self._smooth.get(e.id, (e.x, e.y, e.angle))
            d2 = (sx - x) ** 2 + (sy - y) ** 2
            # 新生成的子体半径较小，为便于调试点击，将命中半径随spawn_progress缩放并设置下限
            scale = config.SPAWN_MIN_SCALE + (1.0 - config.SPAWN_MIN_SCALE) * float(getattr(e, "spawn_progress", 1.0))
            eff_r = max(10.0, e.radius * scale)
            if float(getattr(e, "spawn_progress", 1.0)) < 0.7:
                eff_r += 4.0
            if d2 < best_d2 and d2 <= (eff_r + 6) ** 2:
                best_d2 = d2
                closest = e
        self.selected_id = closest.id if closest else None

    def handle_events(self, world: WorldState):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._pick_entity(world, pygame.mouse.get_pos())
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.show_rays = not self.show_rays
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                elif event.key == pygame.K_LEFTBRACKET:
                    # 前端化改调：加大调整力度并平滑限制
                    self._fov_range_scale = clamp(self._fov_range_scale - 0.15, 0.4, 2.4)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._fov_range_scale = clamp(self._fov_range_scale + 0.15, 0.4, 2.4)
                elif event.key == pygame.K_SLASH:
                    # 切换FOV显示模式：wedge -> lines -> off -> wedge
                    if self._fov_mode == "wedge":
                        self._fov_mode = "lines"
                        self._fov_range_scale = 1.0
                    elif self._fov_mode == "lines":
                        self._fov_mode = "off"
                    else:
                        self._fov_mode = "wedge"
                        self._fov_range_scale = 1.0
                elif event.key == pygame.K_n:
                    # 切换传感器条显示
                    self.show_sensors = not self.show_sensors
                elif event.key == pygame.K_MINUS:
                    self._ray_count_delta = max(-12, self._ray_count_delta - 2)
                elif event.key == pygame.K_EQUALS:
                    self._ray_count_delta = min(24, self._ray_count_delta + 2)
                # 移除主体切换键（仅软体）
                elif event.key == pygame.K_g:
                    # 切换帧回放开关
                    if self._ghost_frames:
                        self._ghost_play = not self._ghost_play
                elif event.key == pygame.K_PERIOD:
                    # 单步前进
                    if self._ghost_frames:
                        self._ghost_play = False
                        self._ghost_idx = min(self._ghost_idx + 1, len(self._ghost_frames) - 1)
                elif event.key == pygame.K_COMMA:
                    # 单步后退
                    if self._ghost_frames:
                        self._ghost_play = False
                        self._ghost_idx = max(self._ghost_idx - 1, 0)

    def draw_world(self, world: WorldState):
        self._draw_grid()
        # 仅移除死亡的捕食者；零能量的猎物仍绘制（原地不动）
        alive = [e for e in world.entities if not (e.type == "hunter" and e.energy <= 0.0)]

        # 若选中实体已不存在（被吃掉或死亡），清空选中，避免点击后看不到属性
        if self.selected_id and not any(e.id == self.selected_id for e in alive):
            self.selected_id = None

        # 事件驱动：先处理事件，再推进成长覆盖（spawn_override）
        self._process_events(world)
        dt = max(0.0, getattr(self, "_last_dt_sec", 0.0))
        for e in alive:
            if e.id in self._spawn_override:
                self._spawn_override[e.id] = min(1.0, self._spawn_override[e.id] + config.SPAWN_GROW_RATE * dt)
        existing_ids = {e.id for e in alive}
        for cid in list(self._spawn_override.keys()):
            if cid not in existing_ids:
                self._spawn_override.pop(cid, None)

        # 视线与目标映射（优先使用射线命中；否则退化为最近实体）
        nearest_map: Dict[str, Tuple[float, float, float]] = {}
        for e in alive:
            fov_deg, fov_range = self._fov_params(e)
            # 默认：沿朝向看向前方
            best_vec: Tuple[float, float, float] = (math.cos(e.angle), math.sin(e.angle), fov_range)

            # 1) 优先使用射线命中：取距离最近的一条命中射线
            hit: Optional[RayHit] = None
            if e.rays:
                hits = [h for h in e.rays if h.hit_id and h.distance > 0.0]
                if hits:
                    hit = min(hits, key=lambda h: h.distance)
            if hit:
                e.target_id = hit.hit_id
                # gaze 以射线方向为主（单位方向），距离为射线距离
                dirx, diry = math.cos(hit.angle), math.sin(hit.angle)
                best_vec = (dirx, diry, min(hit.distance, fov_range))
            else:
                # 2) 回退：最近实体（欧氏距离）
                best_d2 = 1e12
                best_tid: Optional[str] = None
                for o in alive:
                    if o.id == e.id:
                        continue
                    dx = (o.x - e.x)
                    dy = (o.y - e.y)
                    d2 = dx * dx + dy * dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best_tid = o.id
                        best_vec = (dx, dy, math.sqrt(max(0.0, d2)))
                e.target_id = best_tid
            nearest_map[e.id] = best_vec

        # 实体
        for e in alive:
            gaze = nearest_map.get(e.id, (math.cos(e.angle), math.sin(e.angle), e.fov_range))
            self._draw_entity(e, (gaze[0], gaze[1]), gaze[2])
        # 选中实体的传感器条
        if self.selected_id:
            sel = next((x for x in alive if x.id == self.selected_id), None)
            if sel:
                self._draw_sensor_strip(sel)
        # 调试面板
        self._draw_debug_panel(world)

        # 叠加外部动作帧复现（在所有元素之上绘制）
        if self._ghost_frames:
            if self._ghost_play:
                self._ghost_time_accum += self._last_dt_sec
                step = int(self._ghost_time_accum * self._ghost_rate)
                if step > 0:
                    self._ghost_idx = min(self._ghost_idx + step, len(self._ghost_frames) - 1)
                    self._ghost_time_accum = 0.0
            gx, gy, ga = self._ghost_frames[self._ghost_idx]
            # 估算速度：相邻两帧间距 * 播放帧率
            if self._ghost_idx > 0:
                px, py, _ = self._ghost_frames[self._ghost_idx - 1]
            else:
                px, py = gx, gy
            gspeed = math.hypot(gx - px, gy - py) * self._ghost_rate
            ghost_color = (255, 255, 255)
            # 幽灵主体：仅使用软体绘制
            g_r = getattr(config, "DEFAULT_RADIUS", 10.0)
            ge = SimpleNamespace(id="ghost", speed=gspeed, angular_velocity=0.0, radius=g_r)
            self._draw_soft_body(ge, gx, gy, ga, g_r, ghost_color)

    def tick(self) -> int:
        dt = self.clock.tick(60)
        # watchdog：帧耗时异常记录由 update_frame 统一上报
        return dt


# 为其他主程序提供的简易调用函数
def launch_frontend(source: Any) -> None:
    """一行代码启动前端渲染循环（阻塞）。"""
    PygameRenderer.run_with(source)


# 供宿主在运行时加载外部动作帧以复现：
# frames: List[Tuple[x, y, angle]]
def load_ghost_frames(renderer: PygameRenderer, frames: list[Tuple[float, float, float]]):
    if not isinstance(renderer, PygameRenderer):
        return
    if not frames:
        renderer._ghost_frames = None
        renderer._ghost_idx = 0
        renderer._ghost_play = False
        renderer._ghost_time_accum = 0.0
        return
    renderer._ghost_frames = frames
    renderer._ghost_idx = 0
    renderer._ghost_play = True
    renderer._ghost_time_accum = 0.0