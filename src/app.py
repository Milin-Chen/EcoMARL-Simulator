from __future__ import annotations

import time
import pygame

from datasource import MockSource, FileJSONSource
from render import launch_frontend, PygameRenderer


def main():
    # 可切换数据源：Mock 或 JSON 文件
    # source = FileJSONSource(path="runtime/world.json")
    source = MockSource(n_hunters=8, n_prey=24)
    # 使用封装好的前端入口，阻塞式运行
    launch_frontend(source)


if __name__ == "__main__":
    main()
