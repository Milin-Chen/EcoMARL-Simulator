"""
简化训练入口 - 快速开始课程学习
Quick Start Training Entry

这是train_curriculum.py的简化包装，提供最常用的功能
"""

import sys
from train_curriculum import train_stage

def main():
    print("=" * 80)
    print("EcoMARL 快速训练")
    print("=" * 80)
    print()
    print("请选择训练模式:")
    print("  1. Stage 1 - 猎人 vs 静止猎物 (基础)")
    print("  2. Stage 2 - 猎人 vs 脚本猎物 (协作)")
    print("  3. Stage 3 - 猎物训练 (逃跑)")
    print("  4. Stage 4 - 联合微调 (平衡)")
    print("  5. 训练所有阶段 (完整流程)")
    print()

    choice = input("请输入选择 (1-5): ").strip()

    stage_map = {
        "1": "stage1",
        "2": "stage2",
        "3": "stage3",
        "4": "stage4",
    }

    # 询问是否启用HPO
    print()
    enable_hpo_input = input("是否启用HPO增强? (y/N): ").strip().lower()
    enable_hpo = enable_hpo_input in ['y', 'yes']

    # 询问设备
    print()
    device_input = input("训练设备 (auto/cpu/cuda, 默认auto): ").strip().lower()
    device = device_input if device_input in ['cpu', 'cuda'] else 'auto'

    print()
    print("-" * 80)

    if choice in stage_map:
        stage = stage_map[choice]
        print(f"开始训练 {stage}...")
        train_stage(
            stage=stage,
            model_dir="curriculum_models",
            device=device,
            enable_hpo=enable_hpo,
            n_envs=4,
        )
    elif choice == "5":
        print("开始完整的4阶段训练...")
        for i in range(1, 5):
            stage = f"stage{i}"
            print(f"\n>>> 训练 {stage} <<<")
            train_stage(
                stage=stage,
                model_dir="curriculum_models",
                device=device,
                enable_hpo=enable_hpo,
                n_envs=4,
            )
    else:
        print("❌ 无效选择")
        return 1

    print()
    print("=" * 80)
    print("✅ 训练完成!")
    print("=" * 80)
    print()
    print("运行演示:")
    print("  python main.py")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n训练已中断")
        sys.exit(1)
