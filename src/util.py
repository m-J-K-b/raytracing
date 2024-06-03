import math
from typing import Any, List, Tuple

import numpy as np
import pygame as pg


class Vec3(pg.Vector3):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    @classmethod
    def random_unit(self) -> "Vec3":
        return Vec3(
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
        ).normalize()

    def absolute(self) -> "Vec3":
        self.x = abs(self.x)
        self.y = abs(self.y)
        self.z = abs(self.z)
        return self.copy()

    def prod(self, other) -> "Vec3":
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)


def fresnel_reflectivity_coefficient(incident, normal, obj_ior):
    cosI = incident.dot(normal)
    if cosI < 0:
        n1 = 1
        n2 = obj_ior
        normal = -normal
    else:
        n1 = obj_ior
        n2 = 1
        cosI = -cosI
    n = n1 / n2
    sinT2 = n * n * (1.0 - cosI * cosI)
    cosT = (1.0 - sinT2) ** 0.5
    rn = (n1 * cosI - n2 * cosT) / (n1 * cosI + n2 * cosT)
    rt = (n2 * cosI - n1 * cosT) / (n2 * cosI + n2 * cosT)
    rn *= rn
    rt *= rt
    if cosT * cosT < 0:
        return 1
    return (rn + rt) * 0.5


def refract(incident, normal, obj_ior):
    cosI = incident.dot(normal)
    if cosI < 0:
        n1 = 1
        n2 = obj_ior
        normal = -normal
    else:
        n1 = obj_ior
        n2 = 1
        cosI = -cosI
    n = n1 / n2
    sinT2 = n * n * (1.0 - cosI * cosI)
    cosT = (1.0 - sinT2) ** 0.5

    if n == 1:
        return incident
    if cosT * cosT < 0:
        refl = 1
        trans = 0
        return incident.reflect(normal)

    return (n * incident + (n * cosI - cosT) * normal).normalize()


def vec_to_sky_coords(vec: Vec3) -> Tuple[float]:
    return 0.5 + np.arctan2(vec.z, vec.x) / (2 * np.pi), 1 - (
        0.5 + np.arcsin(vec.y) / np.pi
    )


def random_hemisphere_sample(normal: Vec3) -> Vec3:
    nd = Vec3.random_unit()
    if nd.dot(normal) < 0:
        return -nd
    return nd


def quadratic_formula(a: float, b: float, c: float) -> Tuple[float]:
    return (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a), (
        -b + (b**2 - 4 * a * c) ** 0.5
    ) / (2 * a)


def smoothstep(v: float | Any, minv: float | Any, maxv: float | Any) -> float | Any:
    if v < minv:
        return 0
    elif v > maxv:
        return 1

    v = (v - minv) / (maxv - minv)

    return v * v * (3 - 2 * v)


def lerp(v1: float | Any, v2: float | Any, t: float | Any) -> float | Any:
    return v1 + (v2 - v1) * t


def smooth_interpolation(
    v1: float | Any, v2: float | Any, t: float | Any
) -> float | Any:
    return v1 + (v2 - v1) * smoothstep(t, 0, 1)


def exponential_interpolation(
    v1: float | Any, v2: float | Any, t: float | Any, exponent: float = 0.5
) -> float | Any:
    return v1 + (v2 - v1) * min(max(t, 0), 1) ** exponent
