from __future__ import annotations

from src.core.vec3 import Vec3


class Ray:
    def __init__(self, origin: Vec3, direction: Vec3, inside: bool = False) -> None:
        self.origin: Vec3 = origin
        self.direction: Vec3 = direction
        self.inside: bool = inside

    def at(self, d: float) -> Vec3:
        return self.origin + self.direction * d
