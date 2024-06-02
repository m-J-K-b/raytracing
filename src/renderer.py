import random
from threading import Thread
from typing import List

import numpy as np

from src.light import Light
from src.ray import Ray
from src.render_result import RenderResult
from src.render_settings import RenderSettings
from src.scene import Scene
from src.util import (
    Vec3,
    fresnel_reflectivity_coefficient,
    lerp,
    random_hemisphere_sample,
    refract,
)


class Renderer:
    def __init__(self):
        self.scene: Scene = None

        self.render_result: RenderResult = None
        self.render_settings: RenderSettings = None

    def raytrace(self, ray, incoming_light=None, ray_color=None, depth=None):
        incoming_light = Vec3(0) if not incoming_light else incoming_light
        ray_color = Vec3(0) if not ray_color else ray_color
        depth = 0 if not depth else depth

        if depth > self.render_settings.MAX_BOUNCES:
            return Vec3(0)

        hits = self.scene.intersect(ray)
        if not hits:
            return incoming_light + ray_color.prod(self.scene.get_environment(ray))

        if hits[0].obj.material.emission_strength > 0:
            emitted_light = (
                hits[0].obj.material.color * hits[0].obj.material.emission_strength
            )
            incoming_light += ray_color.prod(emitted_light)
        ray_color = ray_color.prod(hits[0].obj.material.color)

        reflection_direction = ray.direction.reflect(hits[0].normal)

        new_ray_direction = lerp(
            random_hemisphere_sample(hits[0].normal),
            reflection_direction,
            hits[0].obj.material.smoothness,
        )

        return self.raytrace(
            Ray(
                ray.origin + new_ray_direction * self.render_settings.EPS,
                new_ray_direction,
            ),
            incoming_light=incoming_light,
            ray_color=ray_color,
            depth=depth + 1,
        )

    def _render_area(self, area: List[float] = [0, 1, 0, 1]) -> None:
        for x in range(
            int(self.render_settings.WIDTH * area[0]),
            int(self.render_settings.WIDTH * area[1]),
        ):
            for y in range(
                int(self.render_settings.HEIGHT * area[2]),
                int(self.render_settings.HEIGHT * area[3]),
            ):
                u, v = (x + 0.5) / self.render_settings.WIDTH * 2 - 1, (
                    y + 0.5
                ) / self.render_settings.HEIGHT * 2 * self.render_settings.ASPECT - self.render_settings.ASPECT
                ray = self.scene.camera.get_ray(
                    u,
                    v,
                )
                self.render_result.add_to_px(
                    x, y, self.raytrace(ray, incoming_light=Vec3(0), ray_color=Vec3(1))
                )

    def _render(self) -> None:
        while self.render_result.rendered_passes < self.render_settings.RENDER_PASSES:
            self.render_result.rendered_passes += 1
            self._render_area(self.render_settings.AREA)
            # np.random.seed(np.random.randint(0, 10000))

        self.render_result.finished = True

    def render(self, render_settings: RenderSettings) -> RenderResult:
        self.init_render(render_settings)
        t = Thread(target=self._render, daemon=True)
        t.start()
        return self.render_result

    def _render_threaded(self, vertical_splits: int, horizontal_splits: int) -> None:
        w = (
            self.render_settings.AREA[1] - self.render_settings.AREA[0]
        ) / vertical_splits
        h = (
            self.render_settings.AREA[3] - self.render_settings.AREA[2]
        ) / horizontal_splits
        while self.render_result.rendered_passes < self.render_settings.RENDER_PASSES:
            self.render_result.rendered_passes += 1
            np.random.seed(np.random.randint(0, 10000))
            threads = []
            for i in range(vertical_splits):
                for j in range(horizontal_splits):
                    area = [
                        i * w + self.render_settings.AREA[0],
                        (i + 1) * w + self.render_settings.AREA[0],
                        j * h + self.render_settings.AREA[2],
                        (j + 1) * h + self.render_settings.AREA[2],
                    ]
                    threads.append(
                        Thread(
                            target=self._render_area,
                            args=(area,),
                            daemon=True,
                        )
                    )

            for t in threads:
                t.start()

            for t in threads:
                t.join()
            self.render_result.save(
                f"./renders/room_{self.render_settings.WIDTH}x{self.render_settings.HEIGHT}_pass#{self.render_result.rendered_passes}.png"
            )
        self.render_result.finished = True

    def render_threaded(
        self,
        render_settings: RenderSettings,
        vertical_splits: int,
        horizontal_splits: int,
    ) -> RenderResult:
        self.init_render(render_settings)
        t = Thread(
            target=(self._render_threaded),
            args=(vertical_splits, horizontal_splits),
            daemon=True,
        )

        t.start()
        return self.render_result

    def init_render(self, render_settings: RenderSettings) -> None:
        self.render_settings = render_settings
        self.render_result = RenderResult(
            render_settings.WIDTH,
            render_settings.HEIGHT,
            self.render_settings.RENDER_PASSES,
        )
        self.scene = render_settings.SCENE
