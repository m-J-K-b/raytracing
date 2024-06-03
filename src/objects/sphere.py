from typing import List

from src.hit_info import HitInfo
from src.material import Material
from src.objects.base import ObjectBase
from src.ray import Ray
from src.util import Vec3, quadratic_formula


class Sphere(ObjectBase):
    def __init__(self, material: Material, origin: Vec3, radius: float) -> None:
        super().__init__(material, origin)
        self.radius: float = radius

    def intersect(self, ray: Ray) -> List[HitInfo]:
        hits = []
        ro = ray.origin - self.origin
        a = ray.direction.dot(ray.direction)
        b = 2 * ro.dot(ray.direction)
        c = ro.dot(ro) - self.radius**2
        radicand = b**2 - 4 * a * c
        if radicand < 0:
            return hits
        t1, t2 = quadratic_formula(a, b, c)
        if t1 > 0:
            hits.append(
                HitInfo(
                    obj=self,
                    depth=t1,
                    normal=(ray.at(t1) - self.origin).normalize(),
                    pos=ray.at(t1),
                )
            )
        if t2 > 0:
            hits.append(
                HitInfo(
                    obj=self,
                    depth=t2,
                    normal=(ray.at(t2) - self.origin).normalize(),
                    pos=ray.at(t2),
                )
            )
        return hits
