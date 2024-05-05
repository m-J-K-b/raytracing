import random
from threading import Thread

import numpy as np
import pygame as pg

from src.post_processing import PostProcessing
from src.scene import Scene
from src.util import Vec3, lerp, random_hemisphere_sample


class Renderer:
    def __init__(self, width, height, passes=1):
        self.WIDTH, self.HEIGHT = width, height
        self.RES = (width, height)
        self.ASPECT = height / width
        self._img_arr = np.zeros(shape=(width, height, 3))
        self.scene = Scene()
        self.max_bounces = 5

        self.break_ = [False]
        self.passes = [0]

        self.post_processing = PostProcessing()

    @property
    def img_arr(self):
        return self._img_arr / (self.passes[0] + 1)

    def reset_image(self):
        self._img_arr = np.zeros(shape=(self.WIDTH, self.HEIGHT, 3))

    def pixel_color(self, ray):
        ray_color = Vec3(1)
        incoming_light = Vec3(0)
        for i in range(self.max_bounces):
            hit_info = self.scene.intersect(ray)
            if hit_info.hit:
                ray.origin = hit_info.hit_pos

                specular_dir = ray.direction.reflect(hit_info.hit_normal)
                diffuse_dir = lerp(
                    random_hemisphere_sample(hit_info.hit_normal),
                    specular_dir,
                    hit_info.hit_obj.material.smoothness,
                ).normalize()
                is_specular = (
                    1
                    if hit_info.hit_obj.material.specular_probability > random.random()
                    else 0
                )

                ray.direction = specular_dir if is_specular else diffuse_dir

                emitted_light = (
                    hit_info.hit_obj.material.emission_color
                    * hit_info.hit_obj.material.emission_strength
                )
                incoming_light += ray_color.prod(emitted_light)
                ray_color = ray_color.prod(
                    (
                        hit_info.hit_obj.material.specular_color
                        if is_specular
                        else hit_info.hit_obj.material.color
                    ),
                )
            else:
                incoming_light += ray_color.prod(self.scene.get_environment(ray))
                break
        return incoming_light

    def _render(self, area=[0, 1, 0, 1]):
        for x in range(int(self.WIDTH * area[0]), int(self.WIDTH * area[1])):
            for y in range(int(self.HEIGHT * area[2]), int(self.HEIGHT * area[3])):
                u, v = (x + 0.5) / self.WIDTH * 2 - 1, (
                    y + 0.5
                ) / self.HEIGHT * 2 * self.ASPECT - self.ASPECT
                ray = self.scene.camera.get_ray(
                    u,
                    v,
                )
                self._img_arr[x, y] += self.pixel_color(ray)
        return self._img_arr

    def render(self, area=[0, 1, 0, 1]):
        t = Thread(target=self._render, args=(area,), daemon=True)
        t.start()

    def _render_threaded(self, grid_width, grid_height):
        w = 1 / grid_width
        h = 1 / grid_height
        self.passes = [0]
        self.break_ = [False]
        while True:
            if not self.break_[0]:
                np.random.seed(np.random.randint(0, 10000))
                threads = []
                for i in range(grid_width):
                    for j in range(grid_height):
                        threads.append(
                            Thread(
                                target=self._render,
                                args=([i * w, (i + 1) * w, j * h, (j + 1) * h],),
                                daemon=True,
                            )
                        )

                for t in threads:
                    t.start()

                for t in threads:
                    t.join()
                self.passes[0] += 1

    def render_threaded(self, grid_width, grid_height):
        t = Thread(
            target=(self._render_threaded), args=(grid_width, grid_height), daemon=True
        )
        t.start()

    def get_img_arr_raw(self):
        return self.img_arr

    def get_img_arr_post_processed(self):
        return self.post_processing.process(self.img_arr)

    def get_img_raw(self):
        return pg.surfarray.make_surface(np.clip(self.img_arr, 0, 1) * 255)

    def get_img_post_processed(self):
        return pg.surfarray.make_surface(self.get_img_arr_post_processed() * 255)
