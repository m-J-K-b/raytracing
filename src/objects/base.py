from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.hit_info import HitInfo

from src.bxdf import BxDF, DiffuseBRDF
from src.material import Material
from src.ray import Ray
from src.util import Vec3


class ObjectBase:
    def __init__(self, material: Material, bxdf: BxDF, origin: Vec3):
        self.material: Material = material
        self.origin = origin
        self.bxdf = bxdf
        self.animations = []

        self.id: int = hash(self)
        self.name: str = "Object"

    def Le(self) -> Vec3:
        return self.material.emission_strength * self.material.color

    def update(self, t) -> None:
        for animation in self.animations:
            animation.update(t)

    def intersect(self, ray: Ray) -> List[HitInfo]:
        raise NotImplementedError()
