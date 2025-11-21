"""
训练日志统计器 - 实时刷新式输出
Training Logger with Real-time Statistical Display
"""

import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class StageStats:
    """阶段统计数据"""
    stage_name: str
    total_steps: int
    start_time: float = field(default_factory=time.time)

    # 训练进度
    current_step: int = 0
    episodes_completed: int = 0

    # 奖励统计
    total_reward_sum: float = 0.0
    episode_rewards: deque = field(default_factory=lambda: deque(maxlen=100))

    # 惩罚计数器 (最近N步)
    penalty_counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    penalty_history: Dict[str, deque] = field(default_factory=dict)

    # 关键事件计数
    predation_count: int = 0  # 捕食次数
    escape_count: int = 0     # 逃脱次数

    # 性能指标
    fps: float = 0.0
    last_update_time: float = field(default_factory=time.time)
    steps_since_update: int = 0


class TrainingLogger:
    """训练日志统计器 - 使用ANSI转义码实现原地刷新"""

    def __init__(self, stage: str, total_steps: int, update_interval: int = 100):
        """
        Args:
            stage: 训练阶段名称
            total_steps: 总训练步数
            update_interval: 统计更新间隔（步数）
        """
        self.stats = StageStats(stage_name=stage, total_steps=total_steps)
        self.update_interval = update_interval
        self.last_display_step = 0

        # 惩罚类型定义（用于显示）
        self.penalty_types = {
            'overlap': '重叠惩罚',
            'jitter': '抖动惩罚',
            'vision_loss': '视野丢失',
            'high_angular': '高角速度',
            'target_switch': '目标切换',
            'stationary': '静止惩罚',
            'herd_conflict': '聚集冲突',
        }

        # 初始化惩罚历史（滑动窗口）
        for penalty_type in self.penalty_types:
            self.stats.penalty_history[penalty_type] = deque(maxlen=1000)

        # 输出初始显示
        self._print_header()

    def _print_header(self):
        """打印表头"""
        print("\n" + "=" * 100)
        print(f"🎯 {self.stats.stage_name} 训练统计")
        print("=" * 100)
        print()  # 预留空行供后续刷新

    def log_step(self, step: int, episode_reward: Optional[float] = None):
        """记录训练步骤"""
        self.stats.current_step = step
        self.stats.steps_since_update += 1

        if episode_reward is not None:
            self.stats.episodes_completed += 1
            self.stats.episode_rewards.append(episode_reward)
            self.stats.total_reward_sum += episode_reward

        # 定期更新显示
        if step - self.last_display_step >= self.update_interval:
            self._update_display()
            self.last_display_step = step

    def log_penalty(self, penalty_type: str, value: float = 1.0):
        """记录惩罚事件

        Args:
            penalty_type: 惩罚类型 ('overlap', 'jitter', 'vision_loss'等)
            value: 惩罚值（默认1表示计数一次）
        """
        if penalty_type in self.penalty_types:
            self.stats.penalty_counters[penalty_type] += 1
            self.stats.penalty_history[penalty_type].append(value)

    def log_event(self, event_type: str):
        """记录关键事件"""
        if event_type == 'predation':
            self.stats.predation_count += 1
        elif event_type == 'escape':
            self.stats.escape_count += 1

    def _update_display(self):
        """更新显示（原地刷新）"""
        # 计算统计数据
        progress = min((self.stats.current_step / self.stats.total_steps) * 100, 100.0)  # 限制最大100%
        elapsed_time = time.time() - self.stats.start_time
        eta = (elapsed_time / max(self.stats.current_step, 1)) * (self.stats.total_steps - self.stats.current_step)
        eta = max(eta, 0)  # 不显示负数ETA

        # 计算FPS
        current_time = time.time()
        time_delta = current_time - self.stats.last_update_time
        if time_delta > 0:
            self.stats.fps = self.stats.steps_since_update / time_delta
        self.stats.last_update_time = current_time
        self.stats.steps_since_update = 0

        # 计算奖励统计
        if self.stats.episode_rewards:
            avg_reward = sum(self.stats.episode_rewards) / len(self.stats.episode_rewards)
            max_reward = max(self.stats.episode_rewards)
            min_reward = min(self.stats.episode_rewards)
            recent_reward = self.stats.episode_rewards[-1] if self.stats.episode_rewards else 0.0
        else:
            avg_reward = max_reward = min_reward = recent_reward = 0.0

        # 计算惩罚统计（最近1000步的频率）
        penalty_stats = {}
        for ptype, pname in self.penalty_types.items():
            count = self.stats.penalty_counters[ptype]
            # 计算最近1000步的频率
            recent_freq = len(self.stats.penalty_history[ptype]) / min(1000, self.stats.current_step) if self.stats.current_step > 0 else 0
            penalty_stats[pname] = (count, recent_freq)

        # 使用ANSI转义码移动光标到固定位置刷新
        # \033[s 保存光标位置
        # \033[u 恢复光标位置
        # \033[K 清除当前行
        # \033[<n>A 向上移动n行

        # 构建输出内容
        output = []
        output.append(f"\033[K📊 进度: {self.stats.current_step:,} / {self.stats.total_steps:,} ({progress:.1f}%)")
        output.append(f"\033[K⏱️  时间: {self._format_time(elapsed_time)} | ETA: {self._format_time(eta)} | FPS: {self.stats.fps:.1f}")
        output.append("\033[K")

        # 奖励统计
        output.append("\033[K🎁 奖励统计:")
        output.append(f"\033[K  · Episode数: {self.stats.episodes_completed}")
        output.append(f"\033[K  · 最近奖励: {recent_reward:+.2f}")
        output.append(f"\033[K  · 平均奖励: {avg_reward:+.2f} (最近100个)")
        output.append(f"\033[K  · 范围: [{min_reward:+.2f}, {max_reward:+.2f}]")
        output.append("\033[K")

        # 惩罚统计
        output.append("\033[K⚠️  惩罚统计 (总计 / 最近频率):")
        for pname, (count, freq) in penalty_stats.items():
            if count > 0:  # 只显示发生过的惩罚
                freq_str = f"{freq*100:.1f}%" if freq > 0 else "0%"
                output.append(f"\033[K  · {pname}: {count:>6} 次 / {freq_str:>6} (最近1000步)")
        output.append("\033[K")

        # 关键事件
        output.append("\033[K🎯 关键事件:")
        output.append(f"\033[K  · 捕食成功: {self.stats.predation_count} 次")
        output.append(f"\033[K  · 逃脱成功: {self.stats.escape_count} 次")
        output.append("\033[K")

        # 动态计算行数
        num_lines = len(output)

        # 向上移动到显示区域起始位置
        sys.stdout.write(f"\033[{num_lines}A")

        # 一次性输出所有行
        sys.stdout.write("\n".join(output) + "\n")
        sys.stdout.flush()

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def finish(self):
        """完成训练，输出最终统计"""
        self._update_display()  # 最后一次更新

        print("\n" + "=" * 100)
        print(f"✅ {self.stats.stage_name} 训练完成!")
        print("=" * 100)

        total_time = time.time() - self.stats.start_time
        avg_reward = sum(self.stats.episode_rewards) / len(self.stats.episode_rewards) if self.stats.episode_rewards else 0

        print(f"总训练步数: {self.stats.current_step:,}")
        print(f"总训练时间: {self._format_time(total_time)}")
        print(f"完成Episodes: {self.stats.episodes_completed}")
        print(f"平均奖励: {avg_reward:+.2f}")
        print(f"捕食成功: {self.stats.predation_count} 次")
        print(f"逃脱成功: {self.stats.escape_count} 次")
        print("=" * 100)
        print()


# ===== 便捷函数 =====

# 全局logger实例
_global_logger: Optional[TrainingLogger] = None


def init_logger(stage: str, total_steps: int, update_interval: int = 100):
    """初始化全局logger"""
    global _global_logger
    _global_logger = TrainingLogger(stage, total_steps, update_interval)
    return _global_logger


def get_logger() -> Optional[TrainingLogger]:
    """获取全局logger"""
    return _global_logger


def log_step(step: int, episode_reward: Optional[float] = None):
    """记录步骤"""
    if _global_logger:
        _global_logger.log_step(step, episode_reward)


def log_penalty(penalty_type: str, value: float = 1.0):
    """记录惩罚"""
    if _global_logger:
        _global_logger.log_penalty(penalty_type, value)


def log_event(event_type: str):
    """记录事件"""
    if _global_logger:
        _global_logger.log_event(event_type)


def finish_logger():
    """完成记录"""
    global _global_logger
    if _global_logger:
        _global_logger.finish()
        _global_logger = None
