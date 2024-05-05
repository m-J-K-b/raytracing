from src.hit_info import HitInfo
from src.objects.object_base import ObjectBase
from src.util import Vec3, quadratic_formula


class Sphere(ObjectBase):
    def __init__(self, material, pos, radius, name=None):
        super().__init__(material, name)
        self.pos = pos
        self.radius = radius

    def intersect(self, ray):
        hit_info = HitInfo()
        ro = ray.origin - self.pos
        a = ray.direction.dot(ray.direction)
        b = 2 * ro.dot(ray.direction)
        c = ro.dot(ro) - self.radius**2
        radicand = b**2 - 4 * a * c
        if radicand < 0:
            hit_info.hit = False
            return hit_info
        hit_distance = min(quadratic_formula(a, b, c))
        if hit_distance > 0:
            hit_info.hit = True
            hit_info.hit_distance = hit_distance
            hit_info.hit_pos = ray.at(hit_distance)
            hit_info.hit_obj = self
            hit_info.hit_normal = self.get_normal(hit_info.hit_pos)
        else:
            hit_info.hit = False
        return hit_info

    def get_normal(self, pos):
        return (pos - self.pos).normalize()
