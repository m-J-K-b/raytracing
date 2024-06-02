import random

import numpy as np

from src.ray import Ray
from src.util import Vec3


class Camera:
    def __init__(
        self,
        pos: Vec3,
        fov: float,
        lookat: Vec3,
        dof_strength: float = 0,
        dof_dist: float = 1,
    ) -> None:
        self.pos: Vec3 = pos
        self.fov: float = fov
        self.d: float = 1 / np.tan(fov / 2)
        self.lookat: Vec3 = lookat
        self.forward: Vec3 = None
        self.right: Vec3 = None
        self.up: Vec3 = None
        self.update_axis()

        self.dof_strength: float = dof_strength
        self.dof_dist: float = dof_dist

    def set_fov(self, fov: float) -> None:
        self.fov = fov
        self.d = 1 / np.tan(fov / 2)

    def set_lookat(self, lookat: Vec3) -> None:
        self.lookat = lookat
        self.update_axis()

    def update_axis(self) -> None:
        self.forward = (self.lookat - self.pos).normalize()
        self.right = Vec3(0, 1, 0).cross(self.forward).normalize()
        self.up = self.forward.cross(self.right).normalize()

    def get_ray(self, u: float, v: float) -> Ray:
        ray_dir = (u * self.right + v * self.up + self.forward * self.d).normalize()
        dof_target = ray_dir * self.dof_dist + self.pos
        ray_pos = (
            self.pos
            + self.right * ((random.random() * 2 - 1) * self.dof_strength)
            + self.up * ((random.random() * 2 - 1) * self.dof_strength)
        )
        ray_dir = (dof_target - ray_pos).normalize()
        return Ray(ray_pos, ray_dir)
