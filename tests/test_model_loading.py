"""测试模型加载功能"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_main_model_loading():
    """测试 main.py 的模型加载逻辑"""
    print("=" * 80)
    print("测试 main.py 模型加载")
    print("=" * 80)

    model_dir = "curriculum_models"

    # 检查模型目录
    if not os.path.exists(model_dir):
        print(f"✗ 模型目录不存在: {model_dir}")
        return False

    print(f"\n✓ 模型目录存在: {model_dir}")

    # 测试猎手模型查找逻辑 (与 main.py 一致)
    print("\n查找猎手模型:")
    hunter_model = None
    for stage in ["stage4", "stage2", "stage1"]:
        path = os.path.join(model_dir, f"{stage}_hunter_final.zip")
        print(f"  检查: {path}")
        if os.path.exists(path):
            hunter_model = path
            print(f"  ✓ 找到: {path}")
            break

    if not hunter_model:
        print("  ✗ 未找到猎手模型")
        return False

    # 测试猎物模型查找逻辑
    print("\n查找猎物模型:")
    prey_model = None
    for stage in ["stage4", "stage3"]:
        path = os.path.join(model_dir, f"{stage}_prey_final.zip")
        print(f"  检查: {path}")
        if os.path.exists(path):
            prey_model = path
            print(f"  ✓ 找到: {path}")
            break

    if not prey_model:
        print("  ✗ 未找到猎物模型")
        return False

    # 测试模型加载
    print("\n测试模型加载:")
    try:
        from stable_baselines3 import PPO

        print(f"\n加载猎手模型: {hunter_model}")
        hunter_ppo = PPO.load(hunter_model, device="cpu")
        print("✓ 猎手模型加载成功")

        print(f"\n加载猎物模型: {prey_model}")
        prey_ppo = PPO.load(prey_model, device="cpu")
        print("✓ 猎物模型加载成功")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试观察空间
    print("\n测试观察空间:")
    try:
        from rl_env.observations import ObservationSpace
        from config import AgentConfig

        obs_extractor = ObservationSpace(AgentConfig())
        print("✓ 观察空间创建成功")

        # 检查观察维度
        info = obs_extractor.get_space_info()
        obs_dim = info["observation_dim"]
        print(f"  观察空间维度: {obs_dim}")

    except Exception as e:
        print(f"✗ 观察空间创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ main.py 模型加载测试通过!")
    print("=" * 80)
    return True


def test_main_rl_model_loading():
    """测试 main_rl.py 的模型加载逻辑"""
    print("\n" + "=" * 80)
    print("测试 main_rl.py 模型加载")
    print("=" * 80)

    # main_rl.py 使用不同的模型格式 (.pt)
    # 它加载 SharedPolicy 而不是 PPO

    # 检查是否有 .pt 模型
    model_dir = "curriculum_models"
    pt_models = list(Path(model_dir).glob("*.pt"))

    if not pt_models:
        print(f"\n⚠ 未找到 .pt 模型文件在 {model_dir}")
        print("  main_rl.py 需要 .pt 格式的 SharedPolicy 模型")
        print("  使用 train_curriculum.py 或 train.py 训练会生成 .zip 模型")
        print("  .zip 模型用于 main.py (PPO模型)")
        print("  如需使用 main_rl.py, 需要导出 .pt 格式的策略网络")
        return False

    print(f"\n✓ 找到 {len(pt_models)} 个 .pt 模型:")
    for model in pt_models:
        print(f"  - {model.name}")

    # 测试加载第一个 .pt 模型
    test_model = pt_models[0]
    print(f"\n测试加载: {test_model}")

    try:
        from rl_env import SharedPolicy, ObservationSpace
        from config import AgentConfig

        # 获取观察维度
        obs_space = ObservationSpace(AgentConfig())
        obs_info = obs_space.get_space_info()
        obs_dim = obs_info["observation_dim"]

        print(f"  观察空间维度: {obs_dim}")

        # 加载策略
        policy = SharedPolicy(obs_dim=obs_dim, action_dim=2, device="cpu")
        policy.load(str(test_model))

        print("✓ 模型加载成功")

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ main_rl.py 模型加载测试通过!")
    print("=" * 80)
    return True


def show_model_summary():
    """显示模型文件摘要"""
    print("\n" + "=" * 80)
    print("模型文件摘要")
    print("=" * 80)

    model_dir = Path("curriculum_models")
    if not model_dir.exists():
        print(f"\n模型目录不存在: {model_dir}")
        return

    # .zip 模型 (PPO格式，用于 main.py)
    zip_models = sorted(model_dir.glob("*.zip"))
    if zip_models:
        print("\n.zip 模型 (PPO格式 - 用于 main.py):")
        for model in zip_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  - {model.name:30s} ({size_mb:.2f} MB)")
    else:
        print("\n.zip 模型: 无")

    # .pt 模型 (SharedPolicy格式，用于 main_rl.py)
    pt_models = sorted(model_dir.glob("*.pt"))
    if pt_models:
        print("\n.pt 模型 (SharedPolicy格式 - 用于 main_rl.py):")
        for model in pt_models:
            size_mb = model.stat().st_size / (1024 * 1024)
            print(f"  - {model.name:30s} ({size_mb:.2f} MB)")
    else:
        print("\n.pt 模型: 无")

    print("\n使用说明:")
    print("  - main.py:    使用 .zip 模型 (Stable-Baselines3 PPO)")
    print("  - main_rl.py: 使用 .pt 模型 (自定义 SharedPolicy)")
    print("\n推荐使用: python main.py")


if __name__ == "__main__":
    print("模型加载测试脚本\n")

    # 显示模型摘要
    show_model_summary()

    # 测试 main.py
    main_ok = test_main_model_loading()

    # 测试 main_rl.py
    main_rl_ok = test_main_rl_model_loading()

    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"  main.py 模型加载:    {'✅ 通过' if main_ok else '❌ 失败'}")
    print(f"  main_rl.py 模型加载: {'✅ 通过' if main_rl_ok else '❌ 失败 (需要.pt模型)'}")
    print()

    if main_ok:
        print("推荐运行命令:")
        print("  python main.py              # 使用训练好的模型")
        print("  python main.py --no-models  # 使用脚本控制")

    sys.exit(0 if (main_ok or main_rl_ok) else 1)
