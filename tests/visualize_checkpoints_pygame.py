"""PyGame可视化 - 显示检查点信息的增强版本"""

import json
import pygame
import sys
from pathlib import Path


def load_world_state(json_path: str = "runtime/world.json"):
    """加载世界状态"""
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def draw_checkpoint_info(screen, world_data, font, font_large):
    """绘制检查点信息面板"""
    if not world_data or "checkpoint_info" not in world_data:
        return

    info = world_data["checkpoint_info"]

    # 面板位置和大小
    panel_x = 10
    panel_y = 10
    panel_width = 350
    panel_height = 140
    padding = 10

    # 绘制半透明背景
    panel_surf = pygame.Surface((panel_width, panel_height))
    panel_surf.set_alpha(200)
    panel_surf.fill((30, 30, 40))
    screen.blit(panel_surf, (panel_x, panel_y))

    # 绘制边框
    pygame.draw.rect(
        screen,
        (100, 150, 200),
        (panel_x, panel_y, panel_width, panel_height),
        2,
    )

    # 标题
    title_text = font_large.render("RL模型检查点", True, (200, 220, 255))
    screen.blit(title_text, (panel_x + padding, panel_y + padding))

    # 检查点信息
    y_offset = panel_y + padding + 30

    # 当前检查点
    checkpoint_text = font.render(
        f"检查点: {info['index']}/{info['total']}", True, (220, 220, 220)
    )
    screen.blit(checkpoint_text, (panel_x + padding, y_offset))
    y_offset += 25

    # 训练步数
    step_text = font.render(
        f"训练步数: {info['step']}", True, (220, 220, 220)
    )
    screen.blit(step_text, (panel_x + padding, y_offset))
    y_offset += 25

    # 猎人奖励
    hunter_reward_text = font.render(
        f"猎人平均奖励: {info['hunter_reward']:.2f}",
        True,
        (255, 100, 100),
    )
    screen.blit(hunter_reward_text, (panel_x + padding, y_offset))
    y_offset += 25

    # 猎物奖励
    prey_reward_text = font.render(
        f"猎物平均奖励: {info['prey_reward']:.2f}",
        True,
        (100, 255, 100),
    )
    screen.blit(prey_reward_text, (panel_x + padding, y_offset))


def draw_controls_help(screen, font_small):
    """绘制控制说明"""
    help_text = [
        "控制说明:",
        "在运行 interactive_demo.py 的终端中:",
        "  数字键 (1,2,3...): 跳转到指定检查点",
        "  n: 下一个检查点",
        "  p: 上一个检查点",
        "  l: 列出所有检查点",
        "  q: 退出",
    ]

    panel_x = 10
    panel_y = 170
    padding = 8
    line_height = 18

    # 计算面板大小
    max_width = max([font_small.size(text)[0] for text in help_text])
    panel_width = max_width + padding * 2
    panel_height = len(help_text) * line_height + padding * 2

    # 绘制半透明背景
    panel_surf = pygame.Surface((panel_width, panel_height))
    panel_surf.set_alpha(180)
    panel_surf.fill((20, 20, 30))
    screen.blit(panel_surf, (panel_x, panel_y))

    # 绘制文本
    y_offset = panel_y + padding
    for text in help_text:
        if text == "控制说明:":
            color = (200, 200, 255)
        else:
            color = (180, 180, 180)

        text_surf = font_small.render(text, True, color)
        screen.blit(text_surf, (panel_x + padding, y_offset))
        y_offset += line_height


def main():
    """主函数 - 增强版PyGame可视化"""
    print("=" * 70)
    print("PyGame检查点可视化")
    print("=" * 70)
    print("\n使用方法:")
    print("1. 在一个终端运行:")
    print("   python interactive_demo.py --run_dir trained_models/ppo_run_XXXXX")
    print("")
    print("2. 在这个终端运行本脚本 (已自动运行)")
    print("")
    print("3. 在 interactive_demo.py 终端中输入命令切换检查点")
    print("=" * 70 + "\n")

    # 初始化Pygame
    pygame.init()

    # 设置窗口
    width, height = 1200, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("EcoMARL - 检查点可视化")

    # 字体
    font = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 18)

    clock = pygame.time.Clock()

    # 主循环
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 加载世界状态
        world_data = load_world_state()

        # 清屏
        screen.fill((30, 30, 35))

        if world_data:
            # 绘制检查点信息
            draw_checkpoint_info(screen, world_data, font, font_large)

            # 绘制控制说明
            draw_controls_help(screen, font_small)

            # 简单的实体显示（可选，这里用简化版本）
            if "entities" in world_data:
                for entity in world_data["entities"]:
                    x = entity.get("x", 0)
                    y = entity.get("y", 0)
                    entity_type = entity.get("type", "prey")

                    # 转换坐标到屏幕空间
                    screen_x = int(x)
                    screen_y = int(y)

                    # 绘制实体
                    color = (255, 100, 100) if entity_type == "hunter" else (100, 255, 100)
                    radius = int(entity.get("radius", 10))

                    if 0 <= screen_x < width and 0 <= screen_y < height:
                        pygame.draw.circle(screen, color, (screen_x, screen_y), radius)

        else:
            # 等待数据
            waiting_text = font_large.render(
                "等待 interactive_demo.py 启动...",
                True,
                (150, 150, 150),
            )
            text_rect = waiting_text.get_rect(center=(width // 2, height // 2))
            screen.blit(waiting_text, text_rect)

            hint_text = font.render(
                "请先运行: python interactive_demo.py --run_dir trained_models/ppo_run_XXXXX",
                True,
                (120, 120, 120),
            )
            hint_rect = hint_text.get_rect(center=(width // 2, height // 2 + 40))
            screen.blit(hint_text, hint_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
