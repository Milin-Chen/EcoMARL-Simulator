#!/usr/bin/env python3
"""
性能基准测试脚本

用于测试不同实体数量下的模拟器性能
"""

import time
import argparse
from core import WorldSimulator
from config import EnvConfig, AgentConfig


def benchmark(n_entities, duration_sec=10, use_parallel=True):
    """
    运行性能基准测试

    Args:
        n_entities: 实体总数
        duration_sec: 测试持续时间(秒)
        use_parallel: 是否使用并行模式
    """
    print(f"\n{'='*60}")
    print(f"性能测试: {n_entities}个实体 ({'并行模式' if use_parallel else '串行模式'})")
    print(f"{'='*60}")

    # 初始化模拟器
    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()

    simulator = WorldSimulator(
        env_config=env_cfg,
        agent_config=agent_cfg,
        use_parallel=use_parallel
    )

    # 分配实体: 1/4猎人, 3/4猎物
    n_hunters = max(1, n_entities // 4)
    n_prey = n_entities - n_hunters

    print(f"初始化: {n_hunters}个猎人, {n_prey}个猎物")
    simulator.initialize(n_hunters=n_hunters, n_prey=n_prey)

    # 预热
    print("预热中...", end="", flush=True)
    for _ in range(10):
        simulator.step()
    print(" 完成")

    # 开始测试
    print(f"运行{duration_sec}秒测试...", end="", flush=True)
    frame_times = []
    start = time.perf_counter()
    frame_count = 0

    while time.perf_counter() - start < duration_sec:
        frame_start = time.perf_counter()
        simulator.step()
        frame_end = time.perf_counter()

        frame_times.append(frame_end - frame_start)
        frame_count += 1

    total_time = time.perf_counter() - start
    print(" 完成")

    # 计算统计数据
    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times)
        min_frame_time = min(frame_times)
        max_frame_time = max(frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

        # 计算百分位数
        sorted_times = sorted(frame_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        print(f"\n结果:")
        print(f"  总帧数: {frame_count}")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  平均FPS: {fps:.1f}")
        print(f"\n帧时间统计:")
        print(f"  平均: {avg_frame_time*1000:.2f}ms")
        print(f"  最快: {min_frame_time*1000:.2f}ms")
        print(f"  最慢: {max_frame_time*1000:.2f}ms")
        print(f"  中位数(P50): {p50*1000:.2f}ms")
        print(f"  P95: {p95*1000:.2f}ms")
        print(f"  P99: {p99*1000:.2f}ms")

        # 性能评级
        if fps >= 55:
            rating = "✅ 优秀"
        elif fps >= 40:
            rating = "✓ 良好"
        elif fps >= 25:
            rating = "⚠ 可接受"
        else:
            rating = "❌ 需要优化"

        print(f"\n性能评级: {rating}")

        # 获取系统统计
        stats = simulator.get_stats()
        if 'raycast_time' in stats:
            print(f"\n系统统计:")
            print(f"  QuadTree构建: {stats.get('quadtree_build_time', 0)*1000:.2f}ms")
            print(f"  射线检测: {stats.get('raycast_time', 0)*1000:.2f}ms")
            print(f"  射线总数: {stats.get('total_rays', 0)}")
            if stats.get('raycast_time', 0) > 0:
                print(f"  射线/秒: {stats.get('total_rays', 0) / stats.get('raycast_time', 1):.0f}")

    simulator.shutdown()
    print(f"{'='*60}\n")

    return fps if frame_times else 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="EcoMARL性能基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试不同规模
  python benchmark_performance.py --scales 50 100 150 200

  # 快速测试
  python benchmark_performance.py --duration 5

  # 对比并行/串行
  python benchmark_performance.py --compare
        """
    )

    parser.add_argument(
        "--scales",
        "-s",
        type=int,
        nargs="+",
        default=[60, 100, 150, 200, 250],
        help="要测试的实体数量列表"
    )

    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=10,
        help="每次测试持续时间(秒)"
    )

    parser.add_argument(
        "--compare",
        "-c",
        action="store_true",
        help="对比并行和串行模式"
    )

    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="禁用并行模式"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("EcoMARL 性能基准测试")
    print("="*60)

    if args.compare:
        # 对比模式
        print("\n📊 并行 vs 串行 对比测试\n")
        results = []

        for n in args.scales:
            print(f"\n测试规模: {n}个实体")
            print("-" * 60)

            # 并行模式
            fps_parallel = benchmark(n, args.duration, use_parallel=True)

            # 串行模式
            fps_serial = benchmark(n, args.duration, use_parallel=False)

            speedup = fps_parallel / fps_serial if fps_serial > 0 else 0
            results.append((n, fps_serial, fps_parallel, speedup))

        # 打印对比表格
        print("\n" + "="*60)
        print("对比结果汇总")
        print("="*60)
        print(f"{'实体数':>8} | {'串行FPS':>10} | {'并行FPS':>10} | {'加速比':>8}")
        print("-" * 60)
        for n, fps_s, fps_p, speedup in results:
            print(f"{n:>8} | {fps_s:>10.1f} | {fps_p:>10.1f} | {speedup:>8.2f}x")
        print("="*60)

    else:
        # 标准测试
        results = []
        for n in args.scales:
            fps = benchmark(n, args.duration, use_parallel=not args.no_parallel)
            results.append((n, fps))

        # 打印结果汇总
        print("\n" + "="*60)
        print("测试结果汇总")
        print("="*60)
        print(f"{'实体数':>8} | {'FPS':>10} | {'评级':>12}")
        print("-" * 60)
        for n, fps in results:
            if fps >= 55:
                rating = "✅ 优秀"
            elif fps >= 40:
                rating = "✓ 良好"
            elif fps >= 25:
                rating = "⚠ 可接受"
            else:
                rating = "❌ 需要优化"
            print(f"{n:>8} | {fps:>10.1f} | {rating:>12}")
        print("="*60)

        # 给出建议
        print("\n💡 性能建议:")
        worst_fps = min(fps for _, fps in results)
        if worst_fps >= 55:
            print("  ✅ 性能表现优秀！系统可以流畅处理当前规模。")
        elif worst_fps >= 40:
            print("  ✓ 性能良好。如需更高FPS，可考虑进一步优化。")
        elif worst_fps >= 25:
            print("  ⚠ 性能可接受，但在高负载下可能出现卡顿。")
            print("     建议: 查看PERFORMANCE_ANALYSIS.md获取优化方案")
        else:
            print("  ❌ 性能需要优化！当前配置无法流畅运行。")
            print("     建议: 立即查看PERFORMANCE_ANALYSIS.md实施优化")
        print()


if __name__ == "__main__":
    main()
