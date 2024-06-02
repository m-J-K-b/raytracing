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


def vec_to_sky_coords(vec: Vec3) -> Tuple[float]:
    return 0.5 + np.arctan2(vec.z, vec.x) / (2 * np.pi), 1 - (
        0.5 + np.arcsin(vec.y) / np.pi
    )


class LightSpectrum:
    SPECTRUM_START_NM = 380
    SPECTRUM_END_NM = 740
    SAMPLE_NUM = SPECTRUM_END_NM - SPECTRUM_START_NM
    NORMALIZATION_CONSTANT = 1 / SAMPLE_NUM

    def __init__(
        self,
    ) -> None:
        self.spectrum = np.linspace(
            self.SPECTRUM_START_NM,
            self.SPECTRUM_END_NM,
            self.SAMPLE_NUM,
            endpoint=True,
            dtype=np.float64,
        )
        self.spectrum_xyz = spectrum_to_xyz(self.spectrum)
        self.radiance = np.ones(shape=(self.SAMPLE_NUM))

    def to_xyz(self):
        return Vec3(
            *(
                np.sum(self.spectrum_xyz * self.radiance[:, None], axis=0)
                * self.NORMALIZATION_CONSTANT
            )
        )

    @property
    def rgb(self):
        pass


def piece_wise_gaussian(
    x: List[int | float] | float | int, m: float | int, t1: float | int, t2: float | int
) -> List[float | int] | float | int:
    return np.exp(-(t1**2) * (x - m) ** 2 / 2) * (x > m) + np.exp(
        -(t2**2) * (x - m) ** 2 / 2
    ) * (x < m)


def spectrum_to_xyz(spectrum):
    x = (
        1.056 * piece_wise_gaussian(spectrum, 599.8, 0.0264, 0.0323)
        + 0.362 * piece_wise_gaussian(spectrum, 442, 0.0624, 0.0374)
        - 0.065 * piece_wise_gaussian(spectrum, 501.1, 0.049, 0.0382)
    )
    y = 0.821 * piece_wise_gaussian(
        spectrum, 568.8, 0.0213, 0.0247
    ) + 0.286 * piece_wise_gaussian(spectrum, 530.9, 0.0613, 0.0322)
    z = 1.217 * piece_wise_gaussian(
        spectrum, 437.0, 0.0845, 0.0278
    ) + 0.681 * piece_wise_gaussian(spectrum, 459.0, 0.0385, 0.0725)
    return np.concatenate((x[:, None], y[:, None], z[:, None]), -1)


def random_hemisphere_sample(normal: Vec3) -> Vec3:
    nd = Vec3.random_unit()
    if nd.dot(normal) < 0:
        return -nd
    return nd


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
