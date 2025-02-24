from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.mesh import ObjectBase

from src.core.vec3 import Vec3


class HitInfo:
    def __init__(
        self,
        obj: object,
        hit: bool = False,
        depth: float = 0.0,
        pos: Vec3 | None = None,
        normal: Vec3 | None = None,
    ):
        self.obj: ObjectBase = obj
        self.hit: bool = hit
        self.depth: float = depth
        self.pos: Vec3 = pos
        self.normal: Vec3 = normal
