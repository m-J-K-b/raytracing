import random

import numpy as np

from src.ray import Ray
from src.util import Vec3


class Camera:
    def __init__(self, pos, fov, lookat, dof_strength=0, dof_dist=1):
        self.pos = pos
        self.fov = fov
        self.forward = (lookat - self.pos).normalize()
        self.right = Vec3(0, 1, 0).cross(self.forward)
        self.up = self.forward.cross(self.right)
        self.d = 1 / np.tan(fov / 2)

        self.dof_strength = dof_strength
        self.dof_dist = dof_dist

    def get_ray(self, u, v):
        ray_dir = (u * self.right + v * self.up + self.forward * self.d).normalize()
        dof_target = ray_dir * self.dof_dist + self.pos
        ray_pos = (
            self.pos
            + self.right * ((random.random() * 2 - 1) * self.dof_strength)
            + self.up * ((random.random() * 2 - 1) * self.dof_strength)
        )
        ray_dir = (dof_target - ray_pos).normalize()
        return Ray(ray_pos, ray_dir)
