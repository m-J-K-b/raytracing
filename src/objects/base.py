from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.hit_info import HitInfo

from src.material import Material
from src.ray import Ray
from src.util import Vec3


class ObjectBase:
    def __init__(self, material: Material, origin: Vec3):
        self.material: Material = material
        self.origin = origin
        self.animations = []

        self.id: int = hash(self)
        self.name: str = "Object"

    def update(self, t) -> None:
        for animation in self.animations:
            animation.update(t)

    def intersect(self, ray: Ray) -> List[HitInfo]:
        raise NotImplementedError()
