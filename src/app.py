from __future__ import annotations

import time
import pygame

from datasource import MockSource, FileJSONSource
from render import PygameRenderer


def main():
    # 可切换数据源：Mock 或 JSON 文件
    # source = FileJSONSource(path="runtime/world.json")
    source = MockSource(n_hunters=8, n_prey=24)

    renderer = PygameRenderer()

    last_frame = None
    while renderer.running:
        # 数据更新
        world = source.poll() or last_frame
        if world is None:
            # 首帧未就绪，等待后端
            time.sleep(0.02)
            continue
        last_frame = world

        # 事件处理
        renderer.handle_events(world)
        if not renderer.paused:
            # 绘制
            renderer.draw_world(world)
            pygame.display.flip()

        renderer.tick()


if __name__ == "__main__":
    main()