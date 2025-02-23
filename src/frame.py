from __future__ import annotations

from src.util import Vec3


class Frame:
    def __init__(
        self,
        local_x=None,
        local_y=None,
        local_z=None,
    ) -> None:
        self.local_x: Vec3 = local_x or Vec3(0)
        self.local_y: Vec3 = local_y or Vec3(0)
        self.local_z: Vec3 = local_z or Vec3(0)

    @classmethod
    def from_normal(cls, normal: Vec3) -> Frame:
        x = normal.cross(Vec3.sample())
        z = x.cross(normal)
        return Frame(x, normal, z)

    def to_local(self, vector: Vec3) -> Vec3:
        return Vec3(
            self.local_x.dot(vector), self.local_y.dot(vector), self.local_z.dot(vector)
        )

    def to_global(self, vector: Vec3) -> Vec3:
        return (
            self.local_x * vector.x + self.local_y * vector.y + self.local_z * vector.z
        )
