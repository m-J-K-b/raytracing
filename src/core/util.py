from __future__ import annotations

from typing import Tuple


def smoothstep(v, minv, maxv):
    if v < minv:
        return 0
    elif v > maxv:
        return 1

    v = (v - minv) / (maxv - minv)

    return v * v * (3 - 2 * v)


def lerp(v1, v2, t):
    return v1 + (v2 - v1) * t


def smooth_interpolation(v1, v2, t):
    return v1 + (v2 - v1) * smoothstep(t, 0, 1)


def exponential_interpolation(v1, v2, t, exponent: float = 0.5):
    return v1 + (v2 - v1) * min(max(t, 0), 1) ** exponent


def quadratic_formula(a: float, b: float, c: float) -> Tuple[float, float]:
    return (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a), (
        -b + (b**2 - 4 * a * c) ** 0.5
    ) / (2 * a)
