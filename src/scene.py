import math
from typing import List

import numpy as np
import pygame as pg

from src.camera import Camera
from src.hit_info import HitInfo
from src.objects.base import ObjectBase
from src.ray import Ray
from src.util import Vec3, vec_to_sky_coords


class Scene:
    def __init__(self) -> None:
        self.objects: List[ObjectBase] = []
        self.camera: Camera = Camera(Vec3(0), np.pi / 3, Vec3(0, 0, 1))
        self.environment_img_arr: np.ndarray = None  # type: ignore
        self.environment_img: np.ndarray = None  # type: ignore

    def add_object(self, obj: ObjectBase) -> None:
        self.objects.append(obj)

    def intersect(self, ray: Ray) -> HitInfo:
        intersections = [
            h for obj in self.objects for h in obj.intersect(ray) if h.depth > 1e-10
        ]
        if intersections:
            return sorted(intersections, key=lambda x: x.depth)[0]
        return intersections

    def set_environment(self, img: np.ndarray | pg.Surface) -> None:
        if isinstance(img, pg.Surface):
            self.environment_img = pg.surfarray.array3d(img) / 255
        self.environment_img = self.environment_img / np.max(self.environment_img)

    def get_environment(self, ray: Ray) -> Vec3:
        if self.environment_img == None:
            return Vec3(0)
        u, v = vec_to_sky_coords(ray.direction)
        if math.isnan(u) or math.isnan(v):
            return Vec3(1)
        c = Vec3(
            self.environment_img[
                int(self.environment_img.shape[0] * u),
                int(self.environment_img.shape[1] * v),
            ]
        )
        return c
