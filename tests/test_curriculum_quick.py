"""快速测试课程学习系统"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AgentConfig, EnvConfig
from rl_env import EnhancedEcoMARLEnv, CurriculumEcoMARLEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

print("="*80)
print("快速测试课程学习系统")
print("="*80)

# 创建环境
print("\n1. 创建Stage1环境...")
agent_config = AgentConfig()
env_config = EnvConfig()

base_env = EnhancedEcoMARLEnv(
    env_config=env_config,
    agent_config=agent_config,
    n_hunters=3,
    n_prey=6,
    max_steps=1000,
    use_v2_rewards=True,
)

curriculum_env = CurriculumEcoMARLEnv(base_env=base_env, stage="stage1")
vec_env = DummyVecEnv([lambda: curriculum_env])

print("✓ 环境创建成功")
print(f"  观察空间: {curriculum_env.observation_space}")
print(f"  动作空间: {curriculum_env.action_space}")

# 测试reset
print("\n2. 测试环境reset...")
obs = vec_env.reset()
print(f"✓ Reset成功，观察形状: {obs.shape}")

# 创建模型
print("\n3. 创建PPO模型...")
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=1e-3,
    n_steps=64,  # 降低以加快测试
    batch_size=32,
    n_epochs=2,
    verbose=1,
    device="cpu",
)
print("✓ 模型创建成功")

# 训练100步
print("\n4. 训练100步...")
model.learn(total_timesteps=100, progress_bar=False)
print("✓ 训练完成!")

# 测试推理
print("\n5. 测试推理...")
obs = vec_env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    print(f"  步{i}: reward={reward[0]:.2f}, done={done[0]}")
    if done[0]:
        obs = vec_env.reset()

print("\n" + "="*80)
print("✓ 所有测试通过! 课程学习系统工作正常。")
print("="*80)
