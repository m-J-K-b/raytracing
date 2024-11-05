from __future__ import annotations

from copy import copy

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
    def default_material(cls) -> Material:
        return Material(
            color=Vec3(1),
            emission_strength=0.0,
            smoothness=0.0,
            transmittance=0.0,
            ior=1.45,
        )

    def copy(self) -> Material:
        return copy(self)
