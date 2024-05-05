import numpy as np

from src.camera import Camera
from src.hit_info import HitInfo
from src.util import Vec3


class Scene:
    def __init__(self) -> None:
        self.objects = []
        self.camera = Camera(Vec3(0), np.pi / 3, Vec3(0, 0, 1))
        self.environment_image = None

    def add_object(self, obj):
        self.objects.append(obj)

    def intersect(self, ray):
        hit_info = HitInfo(hit=False, hit_distance=float("inf"))
        for obj in self.objects:
            hit_info2 = obj.intersect(ray)
            if hit_info2.hit:
                if hit_info2.hit_distance < hit_info.hit_distance:
                    hit_info = hit_info2
        return hit_info

    def set_environment(self, img):
        self.environment_image = img

    def get_environment(self, ray):
        if self.environment_image == None:
            return Vec3(0)
        u, v = 0.5 + np.arctan2(ray.direction.z, ray.direction.x) / (2 * np.pi), 1 - (
            0.5 + np.arcsin(ray.direction.y) / np.pi
        )
        x, y = int(self.environment_image.get_width() * u), int(
            self.environment_image.get_height() * v
        )
        c = Vec3(self.environment_image.get_at((x, y))[0:3]) / 255
        return c
