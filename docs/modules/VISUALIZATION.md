# 可视化系统详解

**Visualization System - PyGame交互式渲染**

---

## 1. PyGame渲染器

### 1.1 功能

**文件**: `frontend/pygame_renderer.py`

- 实时渲染所有实体
- 显示视野范围
- 智能体视角切换
- 性能统计显示
- 暂停/继续控制

### 1.2 启动

```bash
# 标准启动
python main.py

# 不使用模型
python main.py --no-models

# 串行模式（禁用并行）
python main.py serial
```

---

## 2. 交互控制

### 2.1 鼠标

- **点击实体**: 选择智能体，显示其视野
- **点击空白**: 取消选择

### 2.2 键盘

- **空格键**: 暂停/继续
- **ESC**: 退出程序
- **关闭窗口**: 退出程序

---

## 3. 显示元素

### 3.1 实体显示

```python
# 猎人
- 颜色: 红色 (200, 50, 50)
- 形状: 圆形 + 朝向三角形

# 猎物
- 颜色: 蓝色 (50, 100, 200)
- 形状: 圆形 + 朝向三角形
```

### 3.2 视野显示

选中实体后显示：
- **扇形视野**: 半透明黄色区域
- **视野边界**: 虚线
- **视野内目标**: 连线标识

### 3.3 调试信息

屏幕左上角显示：
- FPS（帧率）
- 实体数量（猎人/猎物）
- 平均能量
- 性能统计

---

## 4. 自定义渲染

```python
from frontend import PygameRenderer
from config import RenderConfig

# 创建配置
render_cfg = RenderConfig()
render_cfg.FPS = 30  # 降低帧率
render_cfg.SHOW_FOV = False  # 隐藏视野

# 应用配置
renderer = PygameRenderer(use_parallel=True)
# 渲染配置会自动加载
```

---

## 📚 相关文档

- [核心模块](CORE_MODULES.md) - 底层数据来源
- [配置系统](CONFIGURATION.md) - 渲染参数

---

**可视化，所见即所得！** 🎮
