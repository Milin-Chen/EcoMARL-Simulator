"""评估训练好的模型性能"""

import sys
import os
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from core import WorldSimulator
from config import EnvConfig, AgentConfig
from rl_env.observations import ObservationSpace
import math


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, hunter_model_path=None, prey_model_path=None):
        self.hunter_model = None
        self.prey_model = None

        # 加载模型
        if hunter_model_path and os.path.exists(hunter_model_path):
            print(f"✓ 加载猎人模型: {hunter_model_path}")
            self.hunter_model = PPO.load(hunter_model_path, device="cpu")
        else:
            print(f"⚠ 猎人模型不存在: {hunter_model_path}")

        if prey_model_path and os.path.exists(prey_model_path):
            print(f"✓ 加载猎物模型: {prey_model_path}")
            self.prey_model = PPO.load(prey_model_path, device="cpu")
        else:
            print(f"⚠ 猎物模型不存在: {prey_model_path}")

        # 观察提取器
        self.obs_extractor = ObservationSpace(AgentConfig())

    def get_action(self, entity, all_entities):
        """获取智能体动作"""
        obs = self.obs_extractor.get_observation(entity, all_entities)

        if entity.type == "hunter" and self.hunter_model:
            action, _ = self.hunter_model.predict(obs, deterministic=True)
            return float(action[0]), float(action[1])
        elif entity.type == "prey" and self.prey_model:
            action, _ = self.prey_model.predict(obs, deterministic=True)
            return float(action[0]), float(action[1])
        elif entity.type == "prey":
            # 脚本逃跑
            return self._get_flee_action(entity, all_entities)
        else:
            # 脚本追击
            return self._get_chase_action(entity, all_entities)

    def _get_chase_action(self, hunter, all_entities):
        """追击脚本"""
        closest_prey = None
        min_dist = float('inf')

        for e in all_entities:
            if e.type == "prey":
                dx = e.x - hunter.x
                dy = e.y - hunter.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_prey = e

        if closest_prey is None:
            return 0.0, 0.0

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

    def _get_flee_action(self, prey, all_entities):
        """逃跑脚本"""
        closest_hunter = None
        min_dist = float('inf')

        for e in all_entities:
            if e.type == "hunter":
                dx = e.x - prey.x
                dy = e.y - prey.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_hunter = e

        if closest_hunter is None:
            return 3.0, np.random.uniform(-0.05, 0.05)

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


def evaluate_episode(evaluator, n_hunters=6, n_prey=18, max_steps=1000, verbose=False):
    """评估一个episode"""
    # 配置
    env_cfg = EnvConfig()
    agent_cfg = AgentConfig()

    # 优化参数
    agent_cfg.HUNTER_SPEED_MAX = 50.0
    agent_cfg.HUNTER_ANGULAR_VELOCITY_MAX = 0.15
    agent_cfg.PREY_SPEED_MAX = 45.0
    agent_cfg.PREY_ANGULAR_VELOCITY_MAX = 0.18

    # 初始化模拟器
    simulator = WorldSimulator(env_cfg, agent_cfg, use_parallel=False)
    simulator.initialize(n_hunters=n_hunters, n_prey=n_prey)

    # 统计
    total_predations = 0
    step = 0

    for step in range(max_steps):
        # 应用动作
        for entity in simulator.entities:
            speed_delta, angular_delta = evaluator.get_action(entity, simulator.entities)

            max_speed = 50.0 if entity.type == "hunter" else 45.0
            entity.speed = np.clip(entity.speed + speed_delta, 0.0, max_speed)
            entity.angular_velocity = np.clip(entity.angular_velocity + angular_delta, -0.2, 0.2)

        # 执行步骤
        world = simulator.step()

        # 统计捕食
        predations = [e for e in world.events if e.type == "predation"]
        if predations:
            total_predations += len(predations)
            if verbose:
                for pred in predations:
                    print(f"  步{step}: 捕食! hunter={pred.hunter_id[:8]}, prey={pred.prey_id[:8]}")

        # 检查灭绝
        hunters = [e for e in world.entities if e.type == "hunter"]
        preys = [e for e in world.entities if e.type == "prey"]

        if len(hunters) == 0 or len(preys) == 0:
            if verbose:
                print(f"  步{step}: 灭绝! hunters={len(hunters)}, preys={len(preys)}")
            break

        # 定期输出
        if verbose and step % 100 == 0:
            print(f"  步{step}: hunters={len(hunters)}, preys={len(preys)}, predations={total_predations}")

    simulator.shutdown()

    return {
        'predations': total_predations,
        'steps': step + 1,
        'predation_rate': total_predations / (step + 1) if step > 0 else 0
    }


def main():
    """主评估函数"""
    print("=" * 80)
    print("模型性能评估")
    print("=" * 80)

    # 测试配置
    test_configs = [
        {
            "name": "Stage1 猎人 + 脚本猎物",
            "hunter": "curriculum_models/stage1_hunter_final.zip",
            "prey": None
        },
        {
            "name": "Stage4 完整模型",
            "hunter": "curriculum_models/stage4_hunter_final.zip",
            "prey": "curriculum_models/stage4_prey_final.zip"
        },
        {
            "name": "脚本追击 + 脚本逃跑 (baseline)",
            "hunter": None,
            "prey": None
        }
    ]

    results = []

    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"测试: {config['name']}")
        print(f"{'='*80}")

        # 检查模型是否存在
        if config['hunter'] and not os.path.exists(config['hunter']):
            print(f"⚠ 跳过 (猎人模型不存在)")
            continue
        if config['prey'] and not os.path.exists(config['prey']):
            print(f"⚠ 跳过 (猎物模型不存在)")
            continue

        # 创建评估器
        evaluator = ModelEvaluator(config['hunter'], config['prey'])

        # 运行多个episodes
        n_episodes = 5
        episode_results = []

        print(f"\n运行 {n_episodes} 个episodes...")
        for ep in range(n_episodes):
            result = evaluate_episode(evaluator, n_hunters=6, n_prey=18, max_steps=500, verbose=(ep == 0))
            episode_results.append(result)
            print(f"  Episode {ep+1}: {result['predations']}次捕食, {result['steps']}步, 捕食率={result['predation_rate']*1000:.2f}/1000步")

        # 统计平均结果
        avg_predations = np.mean([r['predations'] for r in episode_results])
        avg_steps = np.mean([r['steps'] for r in episode_results])
        avg_rate = np.mean([r['predation_rate'] for r in episode_results])

        results.append({
            'name': config['name'],
            'avg_predations': avg_predations,
            'avg_steps': avg_steps,
            'avg_rate': avg_rate
        })

        print(f"\n平均结果:")
        print(f"  捕食次数: {avg_predations:.2f}")
        print(f"  存活步数: {avg_steps:.2f}")
        print(f"  捕食率: {avg_rate*1000:.2f} per 1000 steps")

    # 总结对比
    print(f"\n{'='*80}")
    print("总结对比")
    print(f"{'='*80}")
    print(f"{'配置':<30} {'平均捕食':<12} {'平均步数':<12} {'捕食率/1k':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<30} {r['avg_predations']:<12.2f} {r['avg_steps']:<12.2f} {r['avg_rate']*1000:<12.2f}")

    print(f"\n{'='*80}")
    print("评估完成!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
