from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from threading import Event, Thread
from typing import List, Tuple

import numpy as np

from src.core import Ray, Vec3
from src.raytracing.render_result import RenderResult
from src.raytracing.render_settings import RenderSettings
from src.raytracing.scene import Scene
from src.raytracing.util import random_hemisphere_sample, refract, schlick_approximation


class Renderer:
    def __init__(self):
        self.scene: Scene = None  # type: ignore
        self.render_result: RenderResult = None  # type: ignore
        self.render_settings: RenderSettings = None  # type: ignore
        self.stop_event = Event()  # Event to signal threads to stop

    def raytrace(self, ray: Ray, incoming_light=None, ray_color=None, depth=None):
        if incoming_light is None:
            incoming_light = Vec3(0, 0, 0)
        if ray_color is None:
            ray_color = Vec3(1, 1, 1)
        if depth is None:
            depth = 0

        if depth > self.render_settings.MAX_BOUNCES:
            return Vec3(0, 0, 0)

        hits = self.scene.intersect(ray)
        if not hits:
            return incoming_light + ray_color.prod(self.scene.get_environment(ray))

        hit_info = hits[0]
        hit_obj = hit_info.obj

        # Adjust the normal and index-of-refraction if the ray is inside the object.
        ior = 1 / hit_obj.material.ior
        if ray.inside:
            hit_info.normal = -hit_info.normal
            ior = hit_obj.material.ior

        if hit_obj.material.emission_strength > 0:
            emitted_light = hit_obj.material.color * hit_obj.material.emission_strength
            return incoming_light + ray_color.prod(emitted_light)

        ray_color = ray_color.prod(hit_obj.material.color)

        reflection_direction = ray.direction.reflect(hit_info.normal)
        fresnel = schlick_approximation(
            -ray.direction, hit_info.normal, hit_obj.material.ior
        )

        cos_theta = max(hit_info.normal.dot(-ray.direction), 0.0)
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        cannot_refract = ior * sin_theta > 1.0

        if (
            cannot_refract
            or np.random.random() * hit_obj.material.transmittance < fresnel
        ):
            if np.random.random() < hit_obj.material.smoothness:
                new_ray_direction = reflection_direction
            else:
                new_ray_direction = random_hemisphere_sample(hit_info.normal)
        else:
            new_ray_direction = refract(ray.direction, hit_info.normal, ior)
            ray.inside = not ray.inside

        # Recursive call to gather the reflected/refracted light.
        reflected_light = self.raytrace(
            Ray(hit_info.pos, new_ray_direction, inside=ray.inside),
            depth=depth + 1,
        )

        incoming_light += ray_color.prod(reflected_light)
        return incoming_light

    def init_render(self, render_settings: RenderSettings) -> None:
        self.render_settings = render_settings
        self.render_result = RenderResult(
            render_settings.WIDTH,
            render_settings.HEIGHT,
            render_settings.RENDER_PASSES,
        )
        self.scene = render_settings.SCENE

    def render_area(self, area: Tuple[int, int, int, int]) -> None:
        x_start, x_end, y_start, y_end = area
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if self.stop_event.is_set():
                    return  # Exit if stop event is set
                u = (x + 0.5) / self.render_settings.WIDTH * 2 - 1
                v = (
                    (y + 0.5)
                    / self.render_settings.HEIGHT
                    * 2
                    * self.render_settings.ASPECT
                    - self.render_settings.ASPECT
                )
                ray = self.scene.camera.get_ray(u, v)
                color = self.raytrace(ray, incoming_light=Vec3(0), ray_color=Vec3(1))
                self.render_result.add_to_px(x, y, color)

    def divide_area(
        self, vertical_splits: int, horizontal_splits: int
    ) -> List[Tuple[int, int, int, int]]:
        width = self.render_settings.WIDTH
        height = self.render_settings.HEIGHT
        w = width // vertical_splits
        h = height // horizontal_splits
        areas = []
        for i in range(vertical_splits):
            for j in range(horizontal_splits):
                x_start = i * w
                x_end = width if i == vertical_splits - 1 else (i + 1) * w
                y_start = j * h
                y_end = height if j == horizontal_splits - 1 else (j + 1) * h
                areas.append((x_start, x_end, y_start, y_end))
        return areas

    def render(self, vertical_splits: int, horizontal_splits: int) -> None:
        areas = self.divide_area(vertical_splits, horizontal_splits)
        with ThreadPoolExecutor() as executor:
            for _ in range(self.render_settings.RENDER_PASSES):
                executor.map(self.render_area, areas)
                self.render_result.rendered_passes += 1
        self.render_result.finished = True

    def start_executor(self, vertical_splits, horizontal_splits):
        t = Thread(
            target=self.render, args=(vertical_splits, horizontal_splits), daemon=True
        )
        t.start()

    def start_render(
        self,
        render_settings: RenderSettings,
        vertical_splits: int,
        horizontal_splits: int,
    ) -> RenderResult:
        self.init_render(render_settings)
        self.start_executor(vertical_splits, horizontal_splits)
        return self.render_result

    def stop_render(self) -> None:
        self.stop_event.set()  # Signal all threads to stop
