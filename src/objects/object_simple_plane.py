from src.hit_info import HitInfo
from src.objects.object_base import ObjectBase
from src.util import Vec3


class SimplePlane(ObjectBase):
    def __init__(self, material, height, name=None):
        super().__init__(material, name)
        self.height = height

    def intersect(self, ray):
        hit_info = HitInfo()
        y_diff = ray.origin.y - self.height
        if not (y_diff > 0 and ray.direction.y < 0) and not (
            y_diff < 0 and ray.direction.y > 0
        ):
            hit_info.hit = False
            return hit_info

        hit_info.hit = True
        hit_info.hit_distance = abs(
            y_diff / ray.direction.y * ray.direction.xz.magnitude()
        )
        hit_info.hit_normal = Vec3(0, 1, 0)
        hit_info.hit_obj = self
        hit_info.hit_pos = ray.origin + ray.direction * hit_info.hit_distance
        return hit_info

    def get_normal(self, pos):
        return Vec3(0, 1, 0)
