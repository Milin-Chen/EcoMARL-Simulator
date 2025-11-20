"""增强版Gym环境 - 修复训练问题

修复重点:
1. 使用EnhancedRewardFunction (rewards_enhanced)
2. 强化接近目标奖励
3. 方向对齐奖励
4. 惩罚原地转圈和无效运动
5. V2所有特性 (灭绝检测、Early Termination、种群平衡)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from models import WorldState, EntityState
from core import WorldSimulator
from config import EnvConfig, AgentConfig
from ..observations import ObservationSpace
from ..rewards import EnhancedRewardFunction, EnhancedRewardFunctionV2


class EnhancedEcoMARLEnv:
    """
    增强版多智能体强化学习环境 - 解决训练问题

    关键改进:
    - 更强的追击行为奖励 (接近目标有持续奖励)
    - 方向对齐奖励 (朝向目标有奖励)
    - 原地转圈惩罚 (检测并惩罚无效旋转)
    - 无效运动惩罚 (高速移动但远离目标)
    - V2所有特性（智能奖励、灭绝检测、种群平衡）

    V2奖励系统 (推荐):
    - 意图性检测 (只奖励有意的追击/逃跑)
    - 奖励归一化 (防止奖励爆炸)
    - 一致性要求 (连续朝向目标才奖励)
    - 大幅降低随机策略奖励 (V1: +3.66 → V2: -6.07)
    """

    def __init__(
        self,
        env_config: EnvConfig = None,
        agent_config: AgentConfig = None,
        n_hunters: int = 6,
        n_prey: int = 18,
        use_parallel: bool = True,
        max_steps: int = 2000,
        enable_extinction_pruning: bool = True,
        use_v2_rewards: bool = True,  # 新参数: 使用V2奖励函数
    ):
        self.env_config = env_config or EnvConfig()
        self.agent_config = agent_config or AgentConfig()

        # 初始化模拟器
        self.simulator = WorldSimulator(
            self.env_config, self.agent_config, use_parallel=use_parallel
        )

        # 初始化参数
        self.n_hunters = n_hunters
        self.n_prey = n_prey
        self.max_steps = max_steps
        self.enable_extinction_pruning = enable_extinction_pruning
        self.use_v2_rewards = use_v2_rewards

        # 观察和奖励 (可选V1或V2)
        self.obs_space = ObservationSpace(self.agent_config)
        if use_v2_rewards:
            print("使用V2奖励函数 (意图性检测版)")
            self.reward_fn = EnhancedRewardFunctionV2()
        else:
            print("使用V1奖励函数 (原始增强版)")
            self.reward_fn = EnhancedRewardFunction()

        # 状态
        self.world: Optional[WorldState] = None
        self.agent_ids: List[str] = []
        self.step_count: int = 0

        # 统计信息
        self.episode_stats = {
            "total_predations": 0,
            "total_births": 0,
            "total_deaths": 0,
            "hunter_extinctions": 0,
            "prey_extinctions": 0,
            "avg_hunter_reward": 0.0,
            "avg_prey_reward": 0.0,
        }

    def reset(self) -> Dict[str, np.ndarray]:
        """
        重置环境

        Returns:
            observations: 字典 {agent_id: observation}
        """
        # 重置模拟器
        self.simulator.initialize(self.n_hunters, self.n_prey)
        self.world = self.simulator.step()

        # 重置计数器
        self.step_count = 0

        # 重置奖励函数内部状态
        self.reward_fn.reset()

        # 重置统计
        self.episode_stats = {
            "total_predations": 0,
            "total_births": 0,
            "total_deaths": 0,
            "hunter_extinctions": 0,
            "prey_extinctions": 0,
            "avg_hunter_reward": 0.0,
            "avg_prey_reward": 0.0,
        }

        # 获取所有智能体ID
        self.agent_ids = [e.id for e in self.world.entities]

        # 返回初始观察
        return self._get_observations()

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict]:
        """
        执行一步

        Args:
            actions: 字典 {agent_id: action}
                    action = [speed_delta, angular_velocity_delta]

        Returns:
            observations: 字典 {agent_id: observation}
            rewards: 字典 {agent_id: reward}
            dones: 字典 {agent_id: done}
            info: 额外信息（包含灭绝状态）
        """
        self.step_count += 1

        # 应用动作
        self._apply_actions(actions)

        # 执行模拟步骤
        prev_world = self.world
        self.world = self.simulator.step()

        # 计算奖励（包含种群平衡）
        rewards = self._compute_rewards(prev_world, self.world)

        # 检查终止条件（包含灭绝检测）
        dones, extinction_occurred = self._compute_dones()

        # 获取观察
        observations = self._get_observations()

        # 更新统计
        self._update_stats(prev_world, self.world, rewards)

        # 额外信息
        info = {
            "stats": self.simulator.get_stats(),
            "tick": self.world.tick,
            "step_count": self.step_count,
            "extinction_occurred": extinction_occurred,
            "episode_stats": self.episode_stats.copy(),
            "population": {
                "hunters": len([e for e in self.world.entities if e.type == "hunter"]),
                "preys": len([e for e in self.world.entities if e.type == "prey"]),
            },
        }

        return observations, rewards, dones, info

    def _apply_actions(self, actions: Dict[str, np.ndarray]):
        """应用智能体动作"""
        for entity in self.simulator.entities:
            if entity.id in actions:
                action = actions[entity.id]
                # action: [speed_delta, angular_velocity_delta]

                # 速度限制
                max_speed = (
                    self.agent_config.HUNTER_SPEED_MAX
                    if entity.type == "hunter"
                    else self.agent_config.PREY_SPEED_MAX
                )

                entity.speed = np.clip(
                    entity.speed + float(action[0]),
                    0.0,
                    max_speed,
                )

                # 角速度限制
                entity.angular_velocity = np.clip(
                    entity.angular_velocity + float(action[1]),
                    -self.agent_config.HUNTER_ANGULAR_VELOCITY_MAX,
                    self.agent_config.HUNTER_ANGULAR_VELOCITY_MAX,
                )

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取所有智能体的观察"""
        observations = {}
        for entity in self.world.entities:
            observations[entity.id] = self.obs_space.get_observation(
                entity, self.world.entities
            )
        return observations

    def _compute_rewards(
        self, prev_world: WorldState, curr_world: WorldState
    ) -> Dict[str, float]:
        """
        计算奖励（包含种群平衡）
        支持V1和V2两种奖励接口
        """
        rewards = {}

        # 1. 计算个体奖励（Enhanced包含追击、方向、惩罚等）
        for entity in curr_world.entities:
            # V2需要prev_entity参数，V1不需要
            if self.use_v2_rewards:
                prev_entity = next(
                    (e for e in prev_world.entities if e.id == entity.id), None
                )
                if prev_entity is None:
                    # 新生成的实体,使用当前状态作为prev
                    prev_entity = entity

                rewards[entity.id] = self.reward_fn.compute_reward(
                    entity, prev_entity, prev_world, curr_world
                )
            else:
                # V1接口
                rewards[entity.id] = self.reward_fn.compute_reward(
                    entity, prev_world, curr_world
                )

        # 2. 计算种群平衡奖励/惩罚
        hunter_penalty, prey_penalty, _ = self.reward_fn.compute_population_rewards(
            curr_world
        )

        # 3. 应用种群惩罚到所有同类实体
        if hunter_penalty != 0:
            for entity in curr_world.entities:
                if entity.type == "hunter" and entity.id in rewards:
                    rewards[entity.id] += hunter_penalty

        if prey_penalty != 0:
            for entity in curr_world.entities:
                if entity.type == "prey" and entity.id in rewards:
                    rewards[entity.id] += prey_penalty

        return rewards

    def _compute_dones(self) -> Tuple[Dict[str, bool], bool]:
        """
        检查终止条件

        Returns:
            dones: 字典 {agent_id: done}
            extinction_occurred: 是否发生灭绝
        """
        dones = {}
        extinction_occurred = False

        # 1. 检查个体死亡
        for entity in self.world.entities:
            done = entity.energy <= 0
            dones[entity.id] = done

        # 2. 检查灭绝情况（Early Termination）
        if self.enable_extinction_pruning:
            hunters = [e for e in self.world.entities if e.type == "hunter"]
            preys = [e for e in self.world.entities if e.type == "prey"]

            if len(hunters) == 0 or len(preys) == 0:
                # 灭绝发生 - 标记所有实体为done
                extinction_occurred = True
                for entity_id in dones:
                    dones[entity_id] = True

        # 3. 检查最大步数
        if self.step_count >= self.max_steps:
            for entity_id in dones:
                dones[entity_id] = True

        return dones, extinction_occurred

    def _update_stats(
        self, prev_world: WorldState, curr_world: WorldState, rewards: Dict[str, float]
    ):
        """更新统计信息"""
        # 捕食事件
        predation_events = [e for e in curr_world.events if e.type == "predation"]
        self.episode_stats["total_predations"] += len(predation_events)

        # 繁殖事件
        birth_events = [e for e in curr_world.events if e.type == "breed"]
        self.episode_stats["total_births"] += len(birth_events)

        # 死亡统计
        prev_entities = set(e.id for e in prev_world.entities)
        curr_entities = set(e.id for e in curr_world.entities)
        deaths = prev_entities - curr_entities
        self.episode_stats["total_deaths"] += len(deaths)

        # 灭绝统计
        hunters = [e for e in curr_world.entities if e.type == "hunter"]
        preys = [e for e in curr_world.entities if e.type == "prey"]

        if len(hunters) == 0 and self.episode_stats["hunter_extinctions"] == 0:
            self.episode_stats["hunter_extinctions"] = 1

        if len(preys) == 0 and self.episode_stats["prey_extinctions"] == 0:
            self.episode_stats["prey_extinctions"] = 1

        # 平均奖励
        if rewards:
            hunter_rewards = [
                r for eid, r in rewards.items()
                if any(e.id == eid and e.type == "hunter" for e in curr_world.entities)
            ]
            prey_rewards = [
                r for eid, r in rewards.items()
                if any(e.id == eid and e.type == "prey" for e in curr_world.entities)
            ]

            if hunter_rewards:
                self.episode_stats["avg_hunter_reward"] = np.mean(hunter_rewards)
            if prey_rewards:
                self.episode_stats["avg_prey_reward"] = np.mean(prey_rewards)

    def render(self, mode: str = "human"):
        """渲染环境（可选实现）"""
        pass

    def close(self):
        """关闭环境"""
        self.simulator.shutdown()

    def get_observation_space(self) -> Dict[str, int]:
        """获取观察空间维度"""
        return self.obs_space.get_space_info()

    def get_action_space(self) -> Dict[str, Tuple[float, float]]:
        """获取动作空间范围"""
        return {
            "speed_delta": (-10.0, 10.0),
            "angular_velocity_delta": (-0.2, 0.2),
        }

    def set_reward_weights(self, **kwargs):
        """设置奖励权重"""
        self.reward_fn.set_weights(**kwargs)

    def get_episode_summary(self) -> Dict[str, Any]:
        """获取episode统计摘要"""
        return {
            **self.episode_stats,
            "steps": self.step_count,
            "final_population": {
                "hunters": len([e for e in self.world.entities if e.type == "hunter"]),
                "preys": len([e for e in self.world.entities if e.type == "prey"]),
            },
        }
