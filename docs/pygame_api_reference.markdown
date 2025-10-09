PyGame 函数与类调用规范（项目使用版）

本文件汇总本项目中实际用到的 PyGame API（函数、类、常量），并给出调用规范、参数说明与常见属性，便于后续重构与团队协作参考。

—

总体约定

- 坐标系：屏幕左上角为 `(0,0)`，x 向右增、y 向下增。
- 单位：角度多用弧度（渲染显示可转为度），速度以像素/秒，时间以秒。
- 颜色：RGB 三元组 `(r,g,b)`，或 RGBA 四元组 `(r,g,b,a)`（a 为透明度，0-255）。
- 帧提交：每帧调用 `pygame.display.flip()` 刷新显示。

—

显示与 Surface

- `pygame.display.set_mode(size: tuple[int,int], flags: int = 0, depth: int = 0) -> Surface`
  - 作用：创建主显示 Surface（画布）。本项目：`pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))`。
  - 常用：`screen.fill(color)` 清屏；`screen.blit(surface, rect)` 绘制；`screen.get_rect()` 获取屏幕矩形。

- `pygame.display.set_caption(title: str) -> None`
  - 作用：设置窗口标题。示例：`pygame.display.set_caption("智能生态模拟器 - PyGame 前端")`。

- `Surface.fill(color: tuple[int,int,int] | tuple[int,int,int,int]) -> Rect`
  - 作用：用颜色填充整张或指定区域 Surface。项目中用于背景：`screen.fill(config.BG_COLOR)`。

- `Surface.blit(source: Surface, dest: Rect | tuple[int,int]) -> Rect`
  - 作用：将一个 Surface 画到另一个 Surface 上。项目中将旋转后的椭圆 `blit` 到屏幕。

- `Surface.get_rect(**kwargs) -> Rect`
  - 作用：获取与 Surface 同尺寸的定位矩形，支持关键字定位（如 `center=(x,y)`）。

—

时间与时钟

- `pygame.time.Clock() -> Clock`
  - 作用：创建时钟对象控制帧率。
  - 常用属性/方法：
    - `Clock.tick(fps: int) -> int`：限制最大帧率并返回上一帧的毫秒间隔（本项目以 60FPS）。

- `pygame.time.get_ticks() -> int`
  - 作用：获取自 PyGame 初始化以来的毫秒数。项目中用于动态“蠕动”效果的时间参数。

—

事件系统

- `pygame.event.get() -> list[Event]`
  - 作用：拉取当前所有未处理事件。
  - 事件对象常用属性：
    - `event.type`：事件类型（见下）。
    - `event.button`：鼠标按键编号（`MOUSEBUTTONDOWN`）。
    - `event.key`：键值（`KEYDOWN`）。

- 事件类型常量（`pygame` 顶层常量）：
  - `pygame.QUIT`：窗口关闭事件。
  - `pygame.MOUSEBUTTONDOWN`：鼠标按下。
  - `pygame.KEYDOWN`：按键按下。

- 鼠标与键盘辅助：
  - `pygame.mouse.get_pos() -> tuple[int,int]`：当前鼠标坐标。
  - 键位常量（本项目使用）：`pygame.K_p`、`pygame.K_r`、`pygame.K_d`、`pygame.K_LEFTBRACKET`、`pygame.K_RIGHTBRACKET`、`pygame.K_MINUS`、`pygame.K_EQUALS`。

—

绘制与几何图元（返回值一般为 `Rect`，表示绘制区域）

- `pygame.draw.line(surface: Surface, color: Color, start_pos: (x,y), end_pos: (x,y), width: int = 1) -> Rect`
  - 作用：绘制直线。项目中用于背景网格与朝向指示线。

- `pygame.draw.circle(surface: Surface, color: Color, center: (x,y), radius: int, width: int = 0) -> Rect`
  - 作用：绘制圆。项目中用于实体主体与眼睛、瞳孔。

- `pygame.draw.ellipse(surface: Surface, color: Color, rect: RectLike, width: int = 0) -> Rect`
  - 作用：绘制椭圆。项目中用于速度方向拉伸的主体与渐隐残影（在 `SRCALPHA` 临时 Surface 上绘制）。

- `pygame.draw.polygon(surface: Surface, color: Color, points: list[(x,y)], width: int = 0) -> Rect`
  - 作用：绘制多边形。项目中用于 FOV 扇形显示（线框：`width=1`）。

—

变换与透明

- `pygame.transform.rotate(surface: Surface, angle_deg: float) -> Surface`
  - 作用：返回旋转后的新 Surface（不会原地修改）。项目中用于将椭圆按速度方向旋转。

- `pygame.SRCALPHA`（常量）
  - 作用：创建带透明通道的 Surface：`pygame.Surface((w,h), pygame.SRCALPHA)`。
  - 用法：先在透明 Surface 上绘制（支持半透明颜色），再旋转并 `blit` 到屏幕，减少锯齿。

- `pygame.Surface(size: (w,h), flags: int = 0) -> Surface`
  - 作用：创建普通/透明的临时画布；若需透明，结合 `pygame.SRCALPHA`。

—

字体与文本

- `pygame.font.SysFont(name: str, size: int, bold: bool = False, italic: bool = False) -> Font`
  - 作用：创建字体。项目中使用 Menlo 字体：`pygame.font.SysFont("Menlo", config.FONT_SIZE)`。

- `Font.render(text: str, antialias: bool, color: Color, background: Color | None = None) -> Surface`
  - 作用：将文本渲染到 Surface（可直接 `blit` 到屏幕）。
  - 项目中：调试面板逐行渲染文本，并 `blit` 到左上角。

—

矩形与定位

- `pygame.Rect(left: int, top: int, width: int, height: int)` 或 `pygame.Rect((x,y,w,h))`
  - 作用：描述矩形区域，常用于绘制与布局。
  - 常用属性：`rect.left/top/right/bottom/width/height/center/centerx/centery`。
  - 常用构造：`pygame.Rect(int(x - r), int(y - r - 10), int(bar_w), bar_h)`（能量条背景）。

—

显示提交

- `pygame.display.flip() -> None`
  - 作用：提交双缓冲内容到屏幕，一般在每帧绘制结束调用。

—

初始化与退出

- `pygame.init() -> tuple[int,int]`
  - 作用：初始化所有可用的 PyGame 模块。项目中在渲染器构造函数中调用。

- `pygame.quit() -> None`
  - 作用：卸载所有 PyGame 模块（项目当前未显式调用，随进程结束释放）。

—

项目内典型调用模式（摘自当前实现）

1) 初始化与主循环

```python
pygame.init()
screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
pygame.display.set_caption("智能生态模拟器 - PyGame 前端")
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_p:
                pass  # 切换暂停

    # 绘制
    screen.fill(config.BG_COLOR)
    pygame.draw.circle(screen, (255, 255, 255), (100, 100), 20)
    pygame.display.flip()
    clock.tick(60)
```

2) 透明 Surface + 椭圆旋转 + blit

```python
body = pygame.Surface((ell_w + 8, ell_h + 8), pygame.SRCALPHA)
pygame.draw.ellipse(body, color, pygame.Rect(4, 4, ell_w, ell_h))
body_rot = pygame.transform.rotate(body, angle_deg)
body_rect = body_rot.get_rect(center=(int(x), int(y)))
screen.blit(body_rot, body_rect)
```

3) 字体与调试面板文本

```python
font = pygame.font.SysFont("Menlo", config.FONT_SIZE)
img = font.render(text, True, config.DEBUG_PANEL_TEXT)
screen.blit(img, (config.PANEL_MARGIN, y))
```

—

常见注意事项

- 旋转与透明：建议先在 `SRCALPHA` Surface 上绘制图形，再旋转并 `blit`，效果更平滑。
- 事件处理：每帧必须清空事件队列（遍历 `pygame.event.get()`），否则事件会积压。
- 帧率与时间：`Clock.tick(60)` 控制 FPS，同时可用返回的毫秒值计算 `dt`；或在数据源侧固定 `dt=1/60` 保持一致性。
- 命中检测：若渲染使用了位置平滑插值（lerp），点击检测也应使用同样的平滑坐标，避免“点不着”。
- 字体：`SysFont` 需系统存在对应字体；若 Menlo 不可用，建议替换为等宽字体以保持面板布局。