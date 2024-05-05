import numpy as np
import pygame as pg


class Vec3(pg.Vector3):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    def prod(self, other):
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)


def random_hemisphere_sample(normal):
    x, y, z = np.random.randn(3)
    nd = Vec3(x, y, z)
    if nd.dot(normal) < 0:
        return -nd
    return nd


def lerp(v1, v2, t):
    return v1 + (v2 - v1) * t


def quadratic_formula(a, b, c):
    return (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a), (
        -b - (b**2 - 4 * a * c) ** 0.5
    ) / (2 * a)


def smoothstep(v, minv, maxv):
    if v < minv:
        return 0
    elif v > maxv:
        return 1

    v = (v - minv) / (maxv - minv)

    return v * v * (3 - 2 * v)
