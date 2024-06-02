import math
from typing import List

import numpy as np
import pygame as pg

from src.camera import Camera
from src.hit_info import HitInfo
from src.material import Material
from src.objects.base import ObjectBase
from src.objects.sphere import Sphere
from src.ray import Ray
from src.util import Vec3, vec_to_sky_coords


class Scene:
    def __init__(self) -> None:
        self.objects: List[ObjectBase] = []
        self.camera: Camera = Camera(Vec3(0), np.pi / 3, Vec3(0, 0, 1))
        self.environment_image: pg.Surface = None
        self.ambient_light_coefficient = 0.2

    def add_object(self, obj: ObjectBase) -> None:
        self.objects.append(obj)

    def intersect(self, ray: Ray) -> HitInfo:
        hits = [
            h for obj in self.objects for h in obj.intersect(ray) if h.depth > 1e-10
        ]
        sorted_hits = sorted(hits, key=lambda x: x.depth)
        return sorted_hits

    def set_environment(self, img: pg.Surface) -> None:
        self.environment_image = img

    def get_environment(self, ray: Ray) -> Vec3:
        if self.environment_image == None:
            return Vec3(0)
        u, v = vec_to_sky_coords(ray.direction)
        if math.isnan(u) or math.isnan(v):
            return Vec3(1)
        x, y = (
            int(self.environment_image.get_width() * u)
            % self.environment_image.get_width(),
            int(self.environment_image.get_height() * v)
            % self.environment_image.get_height(),
        )
        c = Vec3(self.environment_image.get_at((x, y))[0:3]) / 255
        return c
