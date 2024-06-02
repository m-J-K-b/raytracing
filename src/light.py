from src.util import Vec3


class Light:
    def __init__(self, strength: float) -> None:
        self.strength = strength


class PointLight(Light):
    def __init__(self, strength: float, position: Vec3) -> None:
        super().__init__(strength)
