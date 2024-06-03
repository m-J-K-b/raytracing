from typing import List

from src.scene import Scene


class RenderSettings:
    def __init__(
        self,
        scene: Scene,
        width: int,
        height: int,
        render_passes: int,
        max_bounces: int,
    ) -> None:
        self.WIDTH: int = width
        self.HEIGHT: int = height
        self.ASPECT: float = height / width
        self.RENDER_PASSES: int = render_passes
        self.MAX_BOUNCES: int = max_bounces
        self.SCENE: Scene = scene
        self.AREA: List[float] = [0, 1, 0, 1]
        self.EPS = 0.0
