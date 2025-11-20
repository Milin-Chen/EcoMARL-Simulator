"""自适应任务调度器"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderer import ParallelRenderer


class AdaptiveTaskScheduler:
    """自适应任务调度器 - 根据负载动态调整并行策略"""

    def __init__(self, renderer: ParallelRenderer):
        self.renderer = renderer
        self.history_times = []
        self.max_history = 30
        self.parallel_threshold = 20  # 实体数阈值

    def should_parallelize(self, num_entities: int) -> bool:
        """判断是否应该使用并行计算"""
        # 实体数较少时,串行更快(避免线程开销)
        if num_entities < self.parallel_threshold:
            return False

        # 基于历史性能动态调整
        if len(self.history_times) >= 10:
            avg_time = sum(self.history_times[-10:]) / 10
            # 如果平均处理时间很短,可能串行更快
            if avg_time < 0.001:  # 1ms
                return False

        return True

    def record_time(self, elapsed: float):
        """记录执行时间"""
        self.history_times.append(elapsed)
        if len(self.history_times) > self.max_history:
            self.history_times.pop(0)

    def get_optimal_batch_size(self, num_entities: int) -> int:
        """获取最优批次大小"""
        # 根据实体数量和工作线程数计算
        workers = self.renderer.num_workers
        return max(5, min(20, num_entities // (workers * 2)))
