"""
优化版主程序 - 集成多核多线程四叉树加速
"""

from __future__ import annotations

from datasource import MockSource, FileJSONSource
from render import launch_frontend, PygameRenderer


def main_optimized():
    """使用优化的数据源运行"""
    print("=" * 70)
    print("捕食者-猎物生态模拟 (多线程+四叉树优化版)")
    print("=" * 70)

    # 方案1: 使用Mock数据源(前端独立运行)
    print("\n初始化数据源...")
    source = MockSource(
        n_hunters=12,  # 猎人数量
        n_prey=48,  # 猎物数量
        use_parallel=True,  # 启用并行优化
    )

    # 方案2: 使用JSON文件数据源(与后端对接)
    # source = FileJSONSource(path="runtime/world.json")

    print("\n性能配置:")
    if source.use_parallel:
        print(f"  ✓ 并行模式: 启用")
        print(f"  ✓ 工作线程: {source.renderer.num_workers}")  # type: ignore
        print(f"  ✓ 并行阈值: {source.scheduler.parallel_threshold} 实体")  # type: ignore
    else:
        print(f"  ⚠ 串行模式")

    print("\n运行模拟...")
    print("提示: 关闭窗口或按Ctrl+C退出\n")

    try:
        # 启动前端渲染循环(需要render.py)
        launch_frontend(source)

        # 或者手动运行几帧进行测试
        for i in range(10):
            world = source.poll()
            stats = source.get_performance_stats()

            print(
                f"帧 {world.tick:4d} | "
                f"实体: {len(world.entities):3d} | "
                f"四叉树: {stats.get('total_nodes', 0):3d}节点 | "
                f"射线/秒: {stats.get('rays_per_second', 0):7.0f}"
            )

    except KeyboardInterrupt:
        print("\n\n收到中断信号,正在清理...")
    finally:
        # 清理资源
        if hasattr(source, "shutdown"):
            source.shutdown()
        print("资源已清理,程序退出.")


def main_performance_demo():
    """性能演示模式"""
    import time

    print("=" * 70)
    print("性能演示模式")
    print("=" * 70)

    entity_counts = [20, 40, 60, 80, 100]

    for count in entity_counts:
        print(f"\n测试 {count} 个实体...")

        source = MockSource(
            n_hunters=count // 3, n_prey=count * 2 // 3, use_parallel=True
        )

        # 预热
        for _ in range(5):
            source.poll()

        # 测试
        start = time.perf_counter()
        for _ in range(60):  # 模拟60帧
            source.poll()
        elapsed = time.perf_counter() - start

        fps = 60 / elapsed
        stats = source.get_performance_stats()

        print(f"  FPS: {fps:.1f}")
        print(f"  平均帧时: {elapsed/60*1000:.2f} ms")
        print(f"  射线/秒: {stats.get('rays_per_second', 0):.0f}")
        print(f"  四叉树节点: {stats.get('total_nodes', 0)}")

        source.shutdown()

    print("\n" + "=" * 70)
    print("演示完成!")


def main_comparison():
    """对比模式:并行vs串行"""
    import time

    print("=" * 70)
    print("对比测试: 并行 vs 串行")
    print("=" * 70)

    entity_count = 600
    frames = 100

    print(f"\n配置: {entity_count}个实体, {frames}帧")

    # 测试串行
    print("\n[1/2] 测试串行模式...")
    source_serial = MockSource(
        n_hunters=entity_count // 3, n_prey=entity_count * 2 // 3, use_parallel=False
    )

    start = time.perf_counter()
    for _ in range(frames):
        source_serial.poll()
    time_serial = time.perf_counter() - start

    print(f"  总耗时: {time_serial:.3f}s")
    print(f"  平均帧时: {time_serial/frames*1000:.2f}ms")
    print(f"  FPS: {frames/time_serial:.1f}")

    # 测试并行
    print("\n[2/2] 测试并行模式...")
    source_parallel = MockSource(
        n_hunters=entity_count // 3, n_prey=entity_count * 2 // 3, use_parallel=True
    )

    start = time.perf_counter()
    for _ in range(frames):
        source_parallel.poll()
    time_parallel = time.perf_counter() - start

    stats = source_parallel.get_performance_stats()

    print(f"  总耗时: {time_parallel:.3f}s")
    print(f"  平均帧时: {time_parallel/frames*1000:.2f}ms")
    print(f"  FPS: {frames/time_parallel:.1f}")
    print(f"  工作线程: {stats.get('num_workers', 0)}")
    print(f"  射线/秒: {stats.get('rays_per_second', 0):.0f}")

    # 计算提升
    speedup = time_serial / time_parallel
    improvement = (1 - time_parallel / time_serial) * 100

    print(f"\n{'─'*70}")
    print(f"性能提升: {speedup:.2f}x")
    print(f"耗时减少: {improvement:.1f}%")
    print(f"{'─'*70}")

    # 清理
    source_parallel.shutdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "demo":
            main_performance_demo()
        elif mode == "compare":
            main_comparison()
        else:
            print(f"未知模式: {mode}")
            print("用法:")
            print("  python app_optimized.py         # 正常运行")
            print("  python app_optimized.py demo    # 性能演示")
            print("  python app_optimized.py compare # 对比测试")
    else:
        main_optimized()
