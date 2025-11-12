import time
import config
from models import EntityState


def benchmark_raycast(entities, iterations=100):
    """基准测试：射线检测性能"""
    import math
    from models import RayHit

    total_time = 0
    total_rays = 0

    for _ in range(iterations):
        start = time.perf_counter()

        for e in entities:
            rays = []
            count = config.DEFAULT_RAY_COUNT
            half = math.radians(e.fov_deg or 45.0) / 2.0
            start_angle = e.angle - half
            step = (half * 2) / max(1, count - 1)

            for i in range(count):
                a = start_angle + i * step
                min_dist = e.fov_range or 200.0
                hit_type = None
                hit_id = None

                dx = math.cos(a)
                dy = math.sin(a)

                for o in entities:
                    if o.id == e.id:
                        continue

                    ox = o.x - e.x
                    oy = o.y - e.y
                    proj = ox * dx + oy * dy

                    if proj < 0:
                        continue

                    closest_x = ox - proj * dx
                    closest_y = oy - proj * dy
                    d2 = closest_x**2 + closest_y**2

                    if d2 <= o.radius**2:
                        dist = proj
                        if 0 < dist < min_dist:
                            min_dist = dist
                            hit_type = o.type
                            hit_id = o.id

                rays.append(
                    RayHit(angle=a, distance=min_dist, hit_type=hit_type, hit_id=hit_id)
                )

            total_rays += len(rays)

        elapsed = time.perf_counter() - start
        total_time += elapsed

    avg_time = total_time / iterations
    rays_per_sec = total_rays / total_time

    return {
        "avg_time": avg_time,
        "total_rays": total_rays,
        "rays_per_sec": rays_per_sec,
        "iterations": iterations,
    }


def benchmark_optimized(source, iterations=100):
    """优化版本基准测试"""
    total_time = 0

    for _ in range(iterations):
        start = time.perf_counter()
        source.poll()
        elapsed = time.perf_counter() - start
        total_time += elapsed

    avg_time = total_time / iterations
    stats = source.get_performance_stats()

    return {"avg_time": avg_time, "iterations": iterations, **stats}


def run_benchmark():
    """运行完整的性能对比测试"""
    print("=" * 70)
    print("性能对比测试: 原始版本 vs 优化版本")
    print("=" * 70)

    test_cases = [
        (32, "中等规模"),
        (64, "大规模"),
        (100, "极大规模"),
    ]

    for entity_count, desc in test_cases:
        print(f"\n{'─' * 70}")
        print(f"测试用例: {desc} ({entity_count} 个实体)")
        print(f"{'─' * 70}")

        # 测试原始版本
        print("\n[1/2] 测试原始串行版本...")
        from temp.datasource import MockSource as OriginalMockSource

        original_source = OriginalMockSource(
            n_hunters=entity_count // 3, n_prey=entity_count * 2 // 3
        )

        original_stats = benchmark_raycast(original_source.entities, iterations=50)

        print(f"  平均耗时: {original_stats['avg_time']*1000:.2f} ms")
        print(f"  射线/秒: {original_stats['rays_per_sec']:.0f}")

        # 测试优化版本
        print("\n[2/2] 测试优化版本(并行+四叉树)...")
        from datasource import MockSource as OptimizedMockSource

        optimized_source = OptimizedMockSource(
            n_hunters=entity_count // 3, n_prey=entity_count * 2 // 3, use_parallel=True
        )

        optimized_stats = benchmark_optimized(optimized_source, iterations=50)

        print(f"  平均耗时: {optimized_stats['avg_time']*1000:.2f} ms")
        print(f"  工作线程: {optimized_stats.get('num_workers', 0)}")
        print(f"  四叉树节点数: {optimized_stats.get('total_nodes', 0)}")
        print(f"  四叉树最大深度: {optimized_stats.get('max_depth', 0)}")
        print(f"  射线/秒: {optimized_stats.get('rays_per_second', 0):.0f}")

        # 计算性能提升
        if original_stats["avg_time"] > 0:
            speedup = original_stats["avg_time"] / optimized_stats["avg_time"]
            improvement = (
                1 - optimized_stats["avg_time"] / original_stats["avg_time"]
            ) * 100

            print(f"\n  ✓ 性能提升: {speedup:.2f}x")
            print(f"  ✓ 耗时减少: {improvement:.1f}%")

        # 清理
        optimized_source.shutdown()

    print(f"\n{'=' * 70}")
    print("测试完成!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_benchmark()
