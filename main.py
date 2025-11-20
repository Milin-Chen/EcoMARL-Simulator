"""主入口 - 运行模拟演示"""

import sys
import os
import numpy as np
from core import WorldSimulator
from frontend import PygameRenderer
from config import EnvConfig, AgentConfig, RenderConfig


class DataSourceAdapter:
    """数据源适配器 - 将WorldSimulator适配为前端数据源"""

    def __init__(self, simulator: WorldSimulator):
        self.simulator = simulator

    def poll(self):
        """获取当前世界状态"""
        return self.simulator.step()

    def get_performance_stats(self):
        """获取性能统计"""
        return self.simulator.get_stats()

    def shutdown(self):
        """关闭模拟器"""
        self.simulator.shutdown()


class ModelControlledDataSource:
    """带模型控制的数据源 - 使用训练好的模型控制智能体"""

    def __init__(
        self,
        simulator: WorldSimulator,
        hunter_model_path: str = None,
        prey_model_path: str = None,
    ):
        self.simulator = simulator
        self.hunter_model = None
        self.prey_model = None

        # 加载配置
        self.agent_config = AgentConfig()

        # 尝试加载模型
        if hunter_model_path and os.path.exists(hunter_model_path):
            try:
                from stable_baselines3 import PPO

                self.hunter_model = PPO.load(hunter_model_path, device="cpu")
                print(f"✓ 加载猎人模型: {hunter_model_path}")
            except Exception as e:
                print(f"⚠ 加载猎人模型失败: {e}")

        if prey_model_path and os.path.exists(prey_model_path):
            try:
                from stable_baselines3 import PPO

                self.prey_model = PPO.load(prey_model_path, device="cpu")
                print(f"✓ 加载猎物模型: {prey_model_path}")
            except Exception as e:
                print(f"⚠ 加载猎物模型失败: {e}")

        # 创建观察提取器
        from rl_env.observations import ObservationSpace

        self.obs_extractor = ObservationSpace(self.agent_config)

        print(f"  - 猎人策略: {'训练模型' if self.hunter_model else '脚本追击'}")
        print(f"  - 猎物策略: {'训练模型' if self.prey_model else '脚本逃跑'}")

    def poll(self):
        """执行一步模拟 (应用模型动作)"""
        import math

        # 为每个智能体应用动作
        for entity in self.simulator.entities:
            # 提取观察
            obs = self.obs_extractor.get_observation(entity, self.simulator.entities)

            # 根据类型和模型可用性选择动作
            if entity.type == "hunter" and self.hunter_model is not None:
                # 使用猎人模型
                action, _ = self.hunter_model.predict(obs, deterministic=True)
                speed_delta, angular_delta = float(action[0]), float(action[1])

            elif entity.type == "prey" and self.prey_model is not None:
                # 使用猎物模型
                action, _ = self.prey_model.predict(obs, deterministic=True)
                speed_delta, angular_delta = float(action[0]), float(action[1])

            elif entity.type == "hunter":
                # 脚本追击 (简单版)
                speed_delta, angular_delta = self._get_chase_action(entity)

            else:  # prey without model
                # 脚本逃跑 (简单版)
                speed_delta, angular_delta = self._get_flee_action(entity)

            # 应用动作 (使用配置参数)
            if entity.type == "hunter":
                max_speed = self.agent_config.HUNTER_SPEED_MAX
                max_angular = self.agent_config.HUNTER_ANGULAR_VELOCITY_MAX
            else:
                max_speed = self.agent_config.PREY_SPEED_MAX
                max_angular = self.agent_config.PREY_ANGULAR_VELOCITY_MAX

            entity.speed = np.clip(entity.speed + speed_delta, 0.0, max_speed)
            entity.angular_velocity = np.clip(
                entity.angular_velocity + angular_delta, -max_angular, max_angular
            )

        # 执行物理步骤
        return self.simulator.step()

    def _get_chase_action(self, hunter):
        """简单的追击脚本"""
        import math

        # 找最近的猎物
        closest_prey = None
        min_dist = float("inf")

        for e in self.simulator.entities:
            if e.type == "prey":
                dx = e.x - hunter.x
                dy = e.y - hunter.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_prey = e

        if closest_prey is None:
            return 0.0, 0.0

        # 计算朝向
        dx = closest_prey.x - hunter.x
        dy = closest_prey.y - hunter.y
        target_angle = math.atan2(dy, dx)

        angle_diff = target_angle - hunter.angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        speed_delta = 8.0 if min_dist < 200.0 else 5.0
        angular_delta = np.clip(angle_diff * 0.5, -0.15, 0.15)

        return speed_delta, angular_delta

    def _get_flee_action(self, prey):
        """简单的逃跑脚本"""
        import math

        # 找最近的猎人
        closest_hunter = None
        min_dist = float("inf")

        for e in self.simulator.entities:
            if e.type == "hunter":
                dx = e.x - prey.x
                dy = e.y - prey.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_hunter = e

        if closest_hunter is None:
            return 3.0, np.random.uniform(-0.05, 0.05)

        # 计算逃跑方向 (背离猎人)
        dx = closest_hunter.x - prey.x
        dy = closest_hunter.y - prey.y
        threat_angle = math.atan2(dy, dx)
        flee_angle = threat_angle + math.pi

        angle_diff = flee_angle - prey.angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        speed_delta = 8.0 if min_dist < 150.0 else 3.0
        angular_delta = np.clip(angle_diff * 0.5, -0.15, 0.15)

        return speed_delta, angular_delta

    def get_performance_stats(self):
        """获取性能统计"""
        return self.simulator.get_stats()

    def shutdown(self):
        """关闭模拟器"""
        self.simulator.shutdown()


def main_simulation(use_parallel: bool = True, use_models: bool = True):
    """运行模拟演示"""
    print("=" * 70)
    print("捕食者-猎物生态模拟 (重构版)")
    print("=" * 70)

    # 初始化配置
    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()
    render_cfg = RenderConfig()

    # 使用优化的物理参数 (与训练时一致)
    agent_cfg.HUNTER_SPEED_MAX = 50.0
    agent_cfg.HUNTER_ANGULAR_VELOCITY_MAX = 0.15
    agent_cfg.HUNTER_FOV_DEG = 120.0
    agent_cfg.HUNTER_VIEW_DISTANCE = 250.0

    agent_cfg.PREY_SPEED_MAX = 45.0
    agent_cfg.PREY_ANGULAR_VELOCITY_MAX = 0.18
    agent_cfg.PREY_FOV_DEG = 150.0
    agent_cfg.PREY_VIEW_DISTANCE = 300.0

    # 初始化模拟器
    print("\n初始化模拟器...")
    simulator = WorldSimulator(
        env_config=env_cfg,
        agent_config=agent_cfg,
        use_parallel=use_parallel,
    )
    simulator.initialize(n_hunters=6, n_prey=18)

    # 检查模型并创建数据源
    print("\n智能体控制策略:")
    if use_models:
        # 查找可用的模型
        model_dir = "curriculum_models_hpo"
        hunter_model = None
        prey_model = None

        # 优先使用最终模型
        for stage in ["stage4", "stage2", "stage1"]:
            path = os.path.join(model_dir, f"{stage}_hunter_final.zip")
            if os.path.exists(path):
                hunter_model = path
                break

        for stage in ["stage4", "stage3"]:
            path = os.path.join(model_dir, f"{stage}_prey_final.zip")
            if os.path.exists(path):
                prey_model = path
                break

        # 创建带模型的数据源
        source = ModelControlledDataSource(simulator, hunter_model, prey_model)
    else:
        # 使用默认行为 (随机运动)
        source = DataSourceAdapter(simulator)
        print("  - 使用默认随机运动")

    print("\n性能配置:")
    if use_parallel:
        print(f"  ✓ 并行模式: 启用")
    else:
        print(f"  ⚠ 串行模式")

    print("\n运行模拟...")
    print("提示: 关闭窗口或按Ctrl+C退出\n")

    try:
        # 启动前端渲染（传递use_parallel参数）
        renderer = PygameRenderer(use_parallel=use_parallel)
        renderer.run_loop(source)

    except KeyboardInterrupt:
        print("\n\n收到中断信号,正在清理...")
    finally:
        source.shutdown()
        print("资源已清理,程序退出.")


def main_headless(steps: int = 100, use_parallel: bool = True):
    """无头模式运行(用于性能测试)"""
    print("=" * 70)
    print("无头模式测试")
    print("=" * 70)

    # 初始化模拟器
    simulator = WorldSimulator(use_parallel=use_parallel)
    simulator.initialize(n_hunters=12, n_prey=48)

    import time

    start = time.perf_counter()

    for i in range(steps):
        world = simulator.step()
        if i % 10 == 0:
            stats = simulator.get_stats()
            print(
                f"Step {i:4d} | "
                f"Hunters: {stats.get('hunters', 0):2d} | "
                f"Preys: {stats.get('preys', 0):2d} | "
                f"H_Energy: {stats.get('hunter_avg_energy', 0):5.1f} | "
                f"P_Energy: {stats.get('prey_avg_energy', 0):5.1f}"
            )

    elapsed = time.perf_counter() - start
    fps = steps / elapsed

    print(f"\n{'='*70}")
    print(f"总耗时: {elapsed:.3f}s")
    print(f"平均帧时: {elapsed/steps*1000:.2f}ms")
    print(f"FPS: {fps:.1f}")
    print(f"{'='*70}")

    simulator.shutdown()


if __name__ == "__main__":
    use_parallel = True
    use_models = True
    mode = "normal"

    # 解析命令行参数
    for arg in sys.argv[1:]:
        if arg == "headless":
            mode = "headless"
        elif arg == "serial":
            use_parallel = False
        elif arg == "--no-models":
            use_models = False
        elif arg in ["--help", "-h"]:
            print("用法:")
            print("  python main.py                    # 正常运行 (使用训练模型)")
            print("  python main.py serial             # 串行模式")
            print("  python main.py --no-models        # 不使用模型 (脚本行为)")
            print("  python main.py serial --no-models # 组合选项")
            print("  python main.py headless           # 无头测试")
            sys.exit(0)

    if mode == "headless":
        main_headless(steps=100, use_parallel=use_parallel)
    else:
        main_simulation(use_parallel=use_parallel, use_models=use_models)
