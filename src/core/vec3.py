from __future__ import annotations

import numpy as np
import pygame as pg


class Vec3(pg.Vector3):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    @classmethod
    def random_unit(cls) -> Vec3:
        return Vec3(
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
        ).normalize()

    def absolute(self) -> Vec3:
        self.x = abs(self.x)
        self.y = abs(self.y)
        self.z = abs(self.z)
        return self.copy()

    def prod(self, other) -> Vec3:
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
