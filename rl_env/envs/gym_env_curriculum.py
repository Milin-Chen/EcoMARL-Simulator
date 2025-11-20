"""
课程学习专用Gym环境包装器
Curriculum Learning Gym Environment Wrapper

支持:
1. 阶段1: 猎物静止不动
2. 阶段2: 猎物按脚本逃跑
3. 阶段3: 猎人策略冻结, 猎物学习
4. 阶段4: 联合训练
"""

import math
import numpy as np
from typing import Dict, Tuple
import gymnasium as gym
from gymnasium import spaces

from .gym_env_enhanced import EnhancedEcoMARLEnv
from ..rewards import CurriculumRewardFunction
from core.world import EntityState, WorldState
from config.agent_config import AgentConfig


class CurriculumEcoMARLEnv(gym.Env):
    """课程学习环境包装器 - 符合Gym API"""

    def __init__(
        self,
        base_env: EnhancedEcoMARLEnv,
        stage: str,
    ):
        """
        Args:
            base_env: 基础MARL环境
            stage: "stage1" | "stage2" | "stage3" | "stage4"
        """
        super().__init__()
        self.base_env = base_env
        self.stage = stage

        # 加载配置
        self.agent_config = AgentConfig()

        # 替换奖励函数
        self.base_env.reward_fn = CurriculumRewardFunction(stage)
        # 确保环境知道使用V2接口（4参数）
        self.base_env.use_v2_rewards = True

        # 获取实际的观察空间维度
        obs_space_info = self.base_env.get_observation_space()
        if 'hunter' in obs_space_info:
            obs_dim = obs_space_info['hunter']
        elif 'prey' in obs_space_info:
            obs_dim = obs_space_info['prey']
        else:
            obs_dim = 80  # fallback

        # 定义观察和动作空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间: [speed_delta, angular_velocity_delta] (使用配置参数)
        self.action_space = spaces.Box(
            low=np.array([-self.agent_config.SPEED_DELTA_MAX, -self.agent_config.ANGULAR_DELTA_MAX], dtype=np.float32),
            high=np.array([self.agent_config.SPEED_DELTA_MAX, self.agent_config.ANGULAR_DELTA_MAX], dtype=np.float32),
            dtype=np.float32,
        )

        # 当前episode的智能体列表
        self.current_agents = []
        self.agent_type_map = {}  # agent_id -> type
        self.obs_dim = obs_dim  # 保存维度

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置基础环境
        obs_dict = self.base_env.reset()

        # 记录当前智能体
        self.current_agents = list(obs_dict.keys())
        self.agent_type_map = {
            eid: next(e.type for e in self.base_env.world.entities if e.id == eid)
            for eid in self.current_agents
        }

        # 转换为单一观察 (使用第一个训练中的智能体)
        training_agent = self._get_training_agent()
        if training_agent:
            obs = obs_dict[training_agent]
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        return obs, {}

    def step(self, action: np.ndarray):
        """执行一步"""
        # 构建动作字典
        actions_dict = {}

        for agent_id in self.current_agents:
            agent_type = self.agent_type_map[agent_id]

            if self.stage == "stage1":
                # 阶段1: 猎人学习, 猎物静止
                if agent_type == "hunter":
                    actions_dict[agent_id] = action
                else:  # prey
                    actions_dict[agent_id] = np.array([0.0, 0.0])  # 静止

            elif self.stage == "stage2":
                # 阶段2: 猎人学习, 猎物脚本逃跑
                if agent_type == "hunter":
                    actions_dict[agent_id] = action
                else:  # prey
                    actions_dict[agent_id] = self._get_flee_action(agent_id)

            elif self.stage == "stage3":
                # 阶段3: 猎人冻结, 猎物学习
                if agent_type == "hunter":
                    actions_dict[agent_id] = self._get_frozen_hunter_action(agent_id)
                else:  # prey
                    actions_dict[agent_id] = action

            else:  # stage4
                # 阶段4: 联合训练 - 需要外部多智能体训练器处理
                actions_dict[agent_id] = action

        # 执行一步
        obs_dict, rewards_dict, dones_dict, info = self.base_env.step(actions_dict)

        # 更新智能体列表 (处理死亡)
        self.current_agents = list(obs_dict.keys())
        self.agent_type_map = {
            eid: next(e.type for e in self.base_env.world.entities if e.id == eid)
            for eid in self.current_agents
        }

        # 获取训练中智能体的观察和奖励
        training_agent = self._get_training_agent()
        if training_agent:
            obs = obs_dict.get(training_agent, np.zeros(self.obs_dim, dtype=np.float32))
            reward = rewards_dict.get(training_agent, 0.0)
            done = dones_dict.get(training_agent, False)
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            reward = 0.0
            done = True

        # 检查episode是否结束 (所有智能体都done 或 灭绝)
        terminated = all(dones_dict.values()) if dones_dict else True
        truncated = info.get("extinction_occurred", False)

        return obs, reward, terminated, truncated, info

    def _get_training_agent(self):
        """获取正在训练的智能体ID"""
        if self.stage in ["stage1", "stage2"]:
            # 猎人训练 - 返回第一个猎人
            for aid in self.current_agents:
                if self.agent_type_map[aid] == "hunter":
                    return aid
        elif self.stage == "stage3":
            # 猎物训练 - 返回第一个猎物
            for aid in self.current_agents:
                if self.agent_type_map[aid] == "prey":
                    return aid
        else:  # stage4
            # 联合训练 - 返回第一个智能体
            return self.current_agents[0] if self.current_agents else None

        return None

    def _get_flee_action(self, prey_id: str) -> np.ndarray:
        """获取猎物的脚本逃跑动作"""
        # 找到当前猎物
        prey = next((e for e in self.base_env.world.entities if e.id == prey_id), None)
        if prey is None:
            return np.array([0.0, 0.0])

        # 找到最近的猎人
        closest_hunter = None
        min_dist = float('inf')

        for entity in self.base_env.world.entities:
            if entity.type == "hunter":
                dx = entity.x - prey.x
                dy = entity.y - prey.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_hunter = entity

        if closest_hunter is None:
            # 没有猎人 - 随机游荡
            return np.array([5.0, np.random.uniform(-0.1, 0.1)])

        # 计算逃跑方向 (远离猎人)
        dx = closest_hunter.x - prey.x
        dy = closest_hunter.y - prey.y
        threat_angle = math.atan2(dy, dx)

        # 目标角度 = 威胁角度 + 180度 (背向猎人)
        flee_angle = threat_angle + math.pi

        # 计算角度差
        angle_diff = flee_angle - prey.angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 速度: 危险时加速
        speed_delta = 5.0 if min_dist < 150.0 else 2.0

        # 转向
        angular_delta = np.clip(angle_diff * 0.5, -0.15, 0.15)

        return np.array([speed_delta, angular_delta])

    def _get_frozen_hunter_action(self, hunter_id: str) -> np.ndarray:
        """获取冻结猎人的动作 (阶段3使用训练好的策略)"""
        # TODO: 从加载的模型获取动作
        # 现在先返回简单的追击脚本

        hunter = next((e for e in self.base_env.world.entities if e.id == hunter_id), None)
        if hunter is None:
            return np.array([0.0, 0.0])

        # 找到最近的猎物
        closest_prey = None
        min_dist = float('inf')

        for entity in self.base_env.world.entities:
            if entity.type == "prey":
                dx = entity.x - hunter.x
                dy = entity.y - hunter.y
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_prey = entity

        if closest_prey is None:
            return np.array([0.0, 0.0])

        # 追击
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

        return np.array([speed_delta, angular_delta])

    def render(self):
        """渲染"""
        return self.base_env.render()

    def close(self):
        """关闭"""
        self.base_env.close()
