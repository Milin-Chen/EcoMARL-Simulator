"""
四叉树空间划分加速结构
用于快速查询空间范围内的实体,优化碰撞检测和视野计算
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class AABB:
    """轴对齐包围盒"""

    x: float
    y: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2

    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在AABB内"""
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def intersects(self, other: AABB) -> bool:
        """判断两个AABB是否相交"""
        return not (
            self.right < other.left
            or self.left > other.right
            or self.bottom < other.top
            or self.top > other.bottom
        )

    def intersects_circle(self, cx: float, cy: float, radius: float) -> bool:
        """判断AABB是否与圆相交"""
        # 找到AABB上距离圆心最近的点
        closest_x = max(self.left, min(cx, self.right))
        closest_y = max(self.top, min(cy, self.bottom))

        # 计算距离
        dx = cx - closest_x
        dy = cy - closest_y

        return (dx * dx + dy * dy) <= (radius * radius)


class QuadTreeNode:
    """四叉树节点"""

    MAX_OBJECTS = 8  # 节点最大容纳对象数
    MAX_DEPTH = 6  # 最大深度

    def __init__(self, bounds: AABB, depth: int = 0):
        self.bounds = bounds
        self.depth = depth
        self.objects: List[Tuple[str, float, float, float]] = []  # (id, x, y, radius)
        self.divided = False
        self.children: Optional[List[QuadTreeNode]] = None

    def subdivide(self):
        """细分当前节点为4个子节点"""
        if self.divided:
            return

        x, y = self.bounds.x, self.bounds.y
        w, h = self.bounds.width / 2, self.bounds.height / 2

        self.children = [
            QuadTreeNode(AABB(x, y, w, h), self.depth + 1),  # 左上
            QuadTreeNode(AABB(x + w, y, w, h), self.depth + 1),  # 右上
            QuadTreeNode(AABB(x, y + h, w, h), self.depth + 1),  # 左下
            QuadTreeNode(AABB(x + w, y + h, w, h), self.depth + 1),  # 右下
        ]
        self.divided = True

        # 将当前对象重新分配到子节点
        objects = self.objects
        self.objects = []
        for obj in objects:
            self._insert_to_children(obj)

    def _insert_to_children(self, obj: Tuple[str, float, float, float]):
        """将对象插入到合适的子节点"""
        obj_id, x, y, radius = obj

        # 检查对象与哪些子节点相交
        for child in self.children:
            if child.bounds.intersects_circle(x, y, radius):
                child.insert(obj)

    def insert(self, obj: Tuple[str, float, float, float]) -> bool:
        """插入对象到四叉树"""
        obj_id, x, y, radius = obj

        # 如果对象不在当前节点范围内,返回False
        if not self.bounds.intersects_circle(x, y, radius):
            return False

        # 如果已经细分,插入到子节点
        if self.divided:
            self._insert_to_children(obj)
            return True

        # 如果未达到容量或深度上限,直接插入
        if len(self.objects) < self.MAX_OBJECTS or self.depth >= self.MAX_DEPTH:
            self.objects.append(obj)
            return True

        # 否则细分并重新分配
        self.subdivide()
        self._insert_to_children(obj)
        return True

    def query_range(self, bounds: AABB, result: Set[str]):
        """查询范围内的所有对象ID"""
        if not self.bounds.intersects(bounds):
            return

        # 检查当前节点的对象
        for obj_id, x, y, radius in self.objects:
            if bounds.intersects_circle(x, y, radius):
                result.add(obj_id)

        # 递归查询子节点
        if self.divided:
            for child in self.children:
                child.query_range(bounds, result)

    def query_circle(self, cx: float, cy: float, radius: float, result: Set[str]):
        """查询圆形范围内的所有对象ID"""
        if not self.bounds.intersects_circle(cx, cy, radius):
            return

        # 检查当前节点的对象
        for obj_id, x, y, r in self.objects:
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            if dist_sq <= (radius + r) ** 2:
                result.add(obj_id)

        # 递归查询子节点
        if self.divided:
            for child in self.children:
                child.query_circle(cx, cy, radius, result)

    def query_nearest(
        self, x: float, y: float, max_dist: float
    ) -> Optional[Tuple[str, float]]:
        """查询最近的对象,返回(id, distance)"""
        if not self.bounds.intersects_circle(x, y, max_dist):
            return None

        best_id = None
        best_dist = max_dist

        # 检查当前节点的对象
        for obj_id, ox, oy, radius in self.objects:
            dx = ox - x
            dy = oy - y
            dist = (dx * dx + dy * dy) ** 0.5 - radius
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id

        # 递归查询子节点
        if self.divided:
            for child in self.children:
                result = child.query_nearest(x, y, best_dist)
                if result and result[1] < best_dist:
                    best_id, best_dist = result

        return (best_id, best_dist) if best_id else None


class QuadTree:
    """四叉树容器"""

    def __init__(self, width: float, height: float):
        self.root = QuadTreeNode(AABB(0, 0, width, height))
        self.width = width
        self.height = height

    def clear(self):
        """清空四叉树"""
        self.root = QuadTreeNode(AABB(0, 0, self.width, self.height))

    def insert(self, entity_id: str, x: float, y: float, radius: float):
        """插入实体"""
        self.root.insert((entity_id, x, y, radius))

    def query_range(self, x: float, y: float, width: float, height: float) -> Set[str]:
        """查询矩形范围内的实体ID"""
        result = set()
        self.root.query_range(AABB(x, y, width, height), result)
        return result

    def query_circle(self, cx: float, cy: float, radius: float) -> Set[str]:
        """查询圆形范围内的实体ID"""
        result = set()
        self.root.query_circle(cx, cy, radius, result)
        return result

    def query_nearest(
        self, x: float, y: float, max_dist: float
    ) -> Optional[Tuple[str, float]]:
        """查询最近的实体"""
        return self.root.query_nearest(x, y, max_dist)

    def get_stats(self) -> dict:
        """获取四叉树统计信息"""

        def count_nodes(node: QuadTreeNode) -> Tuple[int, int, int]:
            """返回(节点数, 叶子节点数, 最大深度)"""
            if not node.divided:
                return (1, 1, node.depth)

            total_nodes = 1
            leaf_nodes = 0
            max_depth = node.depth

            for child in node.children:
                n, l, d = count_nodes(child)
                total_nodes += n
                leaf_nodes += l
                max_depth = max(max_depth, d)

            return (total_nodes, leaf_nodes, max_depth)

        nodes, leaves, depth = count_nodes(self.root)
        return {
            "total_nodes": nodes,
            "leaf_nodes": leaves,
            "max_depth": depth,
        }
