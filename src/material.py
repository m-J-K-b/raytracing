from copy import copy

import numpy as np

from src.util import Vec3


class Material:
    def __init__(
        self,
        color: Vec3,
        emission_strength: float,
        smoothness: float,
        transmittance: float,
        ior: float,
    ) -> None:
        self.color: Vec3 = color
        self.emission_strength: float = emission_strength
        self.smoothness: float = smoothness
        self.transmittance: float = transmittance
        self.ior: float = ior

    @classmethod
    def default_material(self) -> "Material":
        return Material(
            color=Vec3(0),
            emission_strength=0,
            smoothness=0,
            is_reflective=False,
            is_refractive=False,
            ior=1,
        )

    def copy(self) -> "Material":
        return copy(self)
