from copy import copy

from src.util import Vec3


class Material:
    def __init__(
        self,
        color,
        emission_color,
        emission_strength,
        smoothness,
        specular_color,
        specular_probability,
    ):
        self.color = color
        self.emission_color = emission_color
        self.emission_strength = emission_strength
        self.smoothness = smoothness
        self.specular_color = specular_color
        self.specular_probability = specular_probability

    @classmethod
    def default_material(self):
        return Material(Vec3(0.7), Vec3(1), 0, 0, Vec3(1), 0)

    def copy(self):
        return copy(self)
