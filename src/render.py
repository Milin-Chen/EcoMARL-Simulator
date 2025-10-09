from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Optional, Tuple

import pygame

import config
from models import WorldState, EntityState, EntityType, RayHit


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class PygameRenderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("智能生态模拟器 - PyGame 前端")
        self.screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", config.FONT_SIZE)

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

        # 可调参数
        self._ray_count_delta = 0
        self._fov_range_scale = 1.0

    def _color_for(self, e: EntityState):
        return config.HUNTER_COLOR if e.type == "hunter" else config.PREY_COLOR

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

    def _draw_stretched_body(self, x: float, y: float, move_a: float, r: float, color: Tuple[int, int, int], speed: float):
        # 圆形沿速度方向拉伸为椭圆，并在反方向绘制渐隐残影椭圆
        base_w = r * 2
        base_h = r * 2
        stretch = min(r * config.STRETCH_MAX_FACTOR, config.STRETCH_SPEED_SCALE * max(0.0, speed))
        ell_w = int(base_w + stretch)
        ell_h = int(max(r * 1.8, base_h - stretch * 0.25))

        # 主体椭圆
        body = pygame.Surface((ell_w + 8, ell_h + 8), pygame.SRCALPHA)
        pygame.draw.ellipse(body, color, pygame.Rect(4, 4, ell_w, ell_h))
        body_rot = pygame.transform.rotate(body, math.degrees(move_a))
        body_rect = body_rot.get_rect(center=(int(x), int(y)))
        self.screen.blit(body_rot, body_rect)

        # 残影椭圆：沿速度反方向依次偏移，透明度逐层降低
        dirx, diry = math.cos(move_a), math.sin(move_a)
        for i in range(1, config.TRAIL_ELLIPSES + 1):
            alpha = int(config.TRAIL_ALPHA_BASE * (config.TRAIL_ALPHA_DECAY ** (i - 1)))
            tw = max(8, int(ell_w - (stretch * 0.25 * i)))
            th = max(8, int(ell_h - (stretch * 0.18 * i)))
            offset = i * config.TRAIL_OFFSET_SCALE * stretch
            cx = x - dirx * offset
            cy = y - diry * offset
            trail_surf = pygame.Surface((tw + 8, th + 8), pygame.SRCALPHA)
            pygame.draw.ellipse(trail_surf, (*color, alpha), pygame.Rect(4, 4, tw, th))
            trail_rot = pygame.transform.rotate(trail_surf, math.degrees(move_a))
            trail_rect = trail_rot.get_rect(center=(int(cx), int(cy)))
            self.screen.blit(trail_rot, trail_rect)

    def _draw_entity(self, e: EntityState, gaze_dir: Tuple[float, float], gaze_dist: float):
        x, y, a = self._lerp_state(e)
        color = self._color_for(e)
        # 平滑分裂缩放：render半径随spawn_progress从小到大
        base_r = e.radius
        scale = config.SPAWN_MIN_SCALE + (1.0 - config.SPAWN_MIN_SCALE) * float(getattr(e, "spawn_progress", 1.0))
        r = base_r * scale
        # 基于运动方向的拉伸角度
        px, py = self._prev_draw.get(e.id, (x, y))
        mvx, mvy = (x - px), (y - py)
        mvlen = math.hypot(mvx, mvy)
        move_a = math.atan2(mvy, mvx) if mvlen > 0.3 else a

        # 拖动式形变主体（沿运动方向拉伸）
        self._draw_stretched_body(x, y, move_a, r, color, e.speed)

        # FOV 扇形（仅选中时显示）
        fov_range = e.fov_range * self._fov_range_scale
        if self.selected_id == e.id:
            half = math.radians(e.fov_deg) / 2
            p1 = (x, y)
            p2 = (x + math.cos(a - half) * fov_range, y + math.sin(a - half) * fov_range)
            p3 = (x + math.cos(a + half) * fov_range, y + math.sin(a + half) * fov_range)
            pygame.draw.polygon(self.screen, (60, 60, 80), [p1, p2, p3], 1)

        # 身体（主圆）
        pygame.draw.circle(self.screen, color, (int(x), int(y)), int(r))

        # 朝向指示
        head = (x + math.cos(a) * r, y + math.sin(a) * r)
        pygame.draw.line(self.screen, (240, 240, 240), (x, y), head, 2)

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
            half = math.radians(e.fov_deg) / 2.0
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

        # 选中高亮
        if self.selected_id == e.id:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(r + 3), 1)
        # 更新前一帧位置
        self._prev_draw[e.id] = (x, y)

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

        sel = next((e for e in world.entities if e.id == self.selected_id), None)
        if sel:
            write(4, f"Selected: {sel.id} ({sel.type})")
            if sel.type == "hunter":
                write(5, f"Energy: {sel.energy:.1f} / split {sel.split_energy:.1f}")
                write(6, f"Speed: {sel.speed:.1f} | AngVel: {sel.angular_velocity:.2f}")
                write(7, f"alive_time: {self._fmt_age(sel.age)} | Gen: {sel.generation}")
                write(8, f"Children: {sel.offspring_count}")
                write(9, f"FOV: {sel.fov_deg:.1f} deg / {sel.fov_range:.1f} px")
                write(10, f"Digestion: {sel.digestion:.2f} | Target: {sel.target_id}")
            else:
                write(5, f"Energy: {sel.energy:.1f} / split {sel.split_energy:.1f}")
                write(6, f"Speed: {sel.speed:.1f}")
                write(7, f"alive_time: {self._fmt_age(sel.age)} | Gen: {sel.generation}")
                write(8, f"Children: {sel.offspring_count}")
                write(9, f"FOV: {sel.fov_deg:.1f} deg / {sel.fov_range:.1f} px")

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
                    self._fov_range_scale = clamp(self._fov_range_scale - 0.1, 0.5, 2.0)
                elif event.key == pygame.K_RIGHTBRACKET:
                    self._fov_range_scale = clamp(self._fov_range_scale + 0.1, 0.5, 2.0)
                elif event.key == pygame.K_MINUS:
                    self._ray_count_delta = max(-12, self._ray_count_delta - 2)
                elif event.key == pygame.K_EQUALS:
                    self._ray_count_delta = min(24, self._ray_count_delta + 2)

    def draw_world(self, world: WorldState):
        self._draw_grid()
        # 仅移除死亡的捕食者；零能量的猎物仍绘制（原地不动）
        alive = [e for e in world.entities if not (e.type == "hunter" and e.energy <= 0.0)]

        # 若选中实体已不存在（被吃掉或死亡），清空选中，避免点击后看不到属性
        if self.selected_id and not any(e.id == self.selected_id for e in alive):
            self.selected_id = None

        # 最近实体映射（用于目光追踪）
        nearest_map: Dict[str, Tuple[float, float, float]] = {}
        for e in alive:
            # 找到最近的其他实体
            best = (math.cos(e.angle), math.sin(e.angle), e.fov_range)  # 默认朝向
            best_d2 = 1e12
            for o in alive:
                if o.id == e.id:
                    continue
                dx = (o.x - e.x)
                dy = (o.y - e.y)
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best = (dx, dy, math.sqrt(max(0.0, d2)))
            nearest_map[e.id] = best

        # 实体
        for e in alive:
            gaze = nearest_map.get(e.id, (math.cos(e.angle), math.sin(e.angle), e.fov_range))
            self._draw_entity(e, (gaze[0], gaze[1]), gaze[2])
        # 调试面板
        self._draw_debug_panel(world)

    def tick(self) -> int:
        return self.clock.tick(60)