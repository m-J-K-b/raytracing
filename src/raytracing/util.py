from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pygame as pg

from src.core import Vec3


def schlick_approximation(incident: Vec3, normal: Vec3, ior: float):
    cos = incident.dot(normal)
    r0 = (1 - ior) / (1 + ior)
    r0 = r0 * r0
    return r0 + (1 - r0) * (1 - cos) ** 5


def refract(incident: Vec3, normal: Vec3, etai_over_etat: float):
    cos_theta = min(normal.dot(-incident), 1)
    r_out_perp = etai_over_etat * (incident + cos_theta * normal)
    r_out_parallel = -np.sqrt(abs(1.0 - r_out_perp.length_squared())) * normal
    return r_out_perp + r_out_parallel


def vec_to_sky_coords(vec: Vec3) -> Tuple[float, float]:
    return 0.5 + np.arctan2(vec.z, vec.x) / (2 * np.pi), 1 - (
        0.5 + np.arcsin(vec.y) / np.pi
    )


def random_hemisphere_sample(normal: Vec3) -> Vec3:
    nd = Vec3.random_unit()
    if nd.dot(normal) < 0:
        return -nd
    return nd
