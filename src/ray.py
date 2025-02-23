from dataclasses import dataclass, field

from src.util import Vec3


@dataclass
class Ray:
    origin: Vec3 = field(default_factory=Vec3)
    direction: Vec3 = field(default_factory=Vec3)
    inside: bool = field(default=False)

    def at(self, d):
        return self.origin + self.direction * d
