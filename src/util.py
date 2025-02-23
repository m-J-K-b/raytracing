from __future__ import annotations

from typing import Any, List, Tuple

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

    @classmethod
    def sample(cls) -> Vec3:
        return Vec3(
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
        )

    def absolute(self) -> Vec3:
        self.x = abs(self.x)
        self.y = abs(self.y)
        self.z = abs(self.z)
        return self.copy()

    def prod(self, other) -> Vec3:
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)


class Vec2(pg.Vector2):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    @classmethod
    def random_unit(cls) -> Vec2:
        return Vec2(
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
        ).normalize()

    @classmethod
    def sample(cls) -> Vec2:
        return Vec2(
            np.random.random() * 2 - 1,
            np.random.random() * 2 - 1,
        )

    def absolute(self) -> Vec2:
        return Vec2(
            abs(self.x),
            abs(self.y),
        )

    def prod(self, other: Vec2) -> Vec2:
        return Vec2(self.x * other.x, self.y * other.y)


def fresnel_conductor(wi: Vec3, eta: complex) -> float:
    """
    Compute the Fresnel reflection for conductors using Snell's law.

    Parameters:
        cos_theta_i (float): Cosine of the angle of incidence.
        eta (complex): Complex refractive index (n - ik).

    Returns:
        float: The Fresnel reflection coefficient.
    """
    cos_theta_i = wi.y
    cos_theta_i = np.clip(cos_theta_i, 0, 1)
    sin2_theta_i = 1 - cos_theta_i**2
    sin2_theta_t = sin2_theta_i / (eta**2)
    cos_theta_t = np.sqrt(1 - sin2_theta_t)
    r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t)
    r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t)

    return (np.abs(r_parl) ** 2 + np.abs(r_perp) ** 2) / 2


def fresnel_dielectric(wi: Vec3, eta: float) -> float:
    """
    Compute the Fresnel reflection for dielectrics using Snell's law.

    Parameters:
        cos_theta_i (float): Cosine of the angle of incidence.
        eta (complex): Complex refractive index (n - ik).

    Returns:
        float: The Fresnel reflection coefficient.
    """
    cos_theta_i = wi.y
    cos_theta_i = np.clip(cos_theta_i, -1, 1)
    if cos_theta_i < 0:
        eta = 1 / eta
        cos_theta_i = -cos_theta_i

    sin2_theta_i = 1 - np.sqrt(cos_theta_i)
    sin2_theta_t = sin2_theta_i / np.sqrt(eta)
    if sin2_theta_t >= 1:
        return 1
    cos_theta_t = np.sqrt(1 - sin2_theta_t)

    r_parl = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t)
    r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t)
    return (np.sqrt(r_parl) + np.sqrt(r_perp)) / 2


def refract(wi: Vec3, normal: Vec3, eta: float) -> Vec3:
    """Calculate the refracted direction of a ray when it enters a medium

    Args:
        wi (Vec3): Ray direction, outwards facing from the object and intersection point
        normal (Vec3): Surface normal, outwards facing from the object and intersection point
        eta (float): ratio of the index of refraction of the current medium and the one the ray will enter

    Returns:
        Vec3: Refracted direction, pointing into the object, away from the normal
    """
    cos_theta_i = normal.dot(wi)
    if cos_theta_i < 0:
        eta = 1 / eta
        cos_theta_i = -cos_theta_i
        normal = -normal

    sin2_theta_i = max(0, 1 - np.sqrt(cos_theta_i))
    sin2_theta_t = sin2_theta_i / np.sqrt(eta)
    if sin2_theta_t >= 1:
        return None

    cos_theta_t = np.sqrt(1 - sin2_theta_t)

    return -wi / eta + (cos_theta_i / eta - cos_theta_t) * normal


def vec_to_sky_coords(vec: Vec3) -> Tuple[float, float]:
    return 0.5 + np.arctan2(vec.z, vec.x) / (2 * np.pi), 1 - (
        0.5 + np.arcsin(vec.y) / np.pi
    )


def sample_uniform_disk_concentric() -> Vec2:
    u = Vec2.sample()
    if u.x == 0 and u.y == 0:
        return Vec2()

    if abs(u.x) > abs(u.y):
        r = u.x
        theta = np.pi / 4 * (u.y / u.x)
    else:
        r = u.y
        theta = np.pi / 2 - np.pi / 4 * (u.x / u.y)

    return Vec2(r * np.cos(theta), r * np.sin(theta))


def sample_cosine_hemisphere() -> Vec3:
    d = sample_uniform_disk_concentric()
    y = np.sqrt(1 - d[0] ** 2 - d[1] ** 2)
    return Vec3(d.x, y, d.y)


def cosine_hemisphere_pdf(sample: Vec3) -> float:
    return sample.y / np.pi


def sample_hemisphere(normal: Vec3) -> Vec3:
    nd = Vec3.random_unit()
    if nd.dot(normal) < 0:
        return -nd
    return nd


def hemisphere_pdf(sample: Vec3) -> float:
    return 1 / (2 * np.pi)


def quadratic_formula(a: float, b: float, c: float) -> Tuple[float, float]:
    return (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a), (
        -b + (b**2 - 4 * a * c) ** 0.5
    ) / (2 * a)


def smoothstep(v: Any, minv: Any, maxv: Any) -> Any:
    if v < minv:
        return 0
    elif v > maxv:
        return 1

    v = (v - minv) / (maxv - minv)

    return v * v * (3 - 2 * v)


def lerp(v1: Any, v2: Any, t: Any) -> Any:
    return v1 + (v2 - v1) * t


def smooth_interpolation(v1: Any, v2: Any, t: Any) -> Any:
    return v1 + (v2 - v1) * smoothstep(t, 0, 1)


def exponential_interpolation(v1: Any, v2: Any, t: Any, exponent: float = 0.5) -> Any:
    return v1 + (v2 - v1) * min(max(t, 0), 1) ** exponent
