from __future__ import annotations

from typing import TYPE_CHECKING, List

from src.hit_info import HitInfo
from src.material import Material
from src.ray import Ray
from src.util import Vec3


class ObjectBase:
    def __init__(self, material: Material, origin: Vec3):
        self.material: Material = material
        self.origin = origin

    def update(self, t) -> None:
        for animation in self.animations:
            animation.update(t)

    def intersect(self, ray: Ray) -> List[HitInfo]:
        raise NotImplementedError()
