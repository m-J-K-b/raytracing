from dataclasses import dataclass

from src.util import Vec3


@dataclass
class Ray:
    origin: Vec3 = None
    direction: Vec3 = None

    def at(self, d):
        return self.origin + self.direction * d
