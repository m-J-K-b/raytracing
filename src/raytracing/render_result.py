import numpy as np
import pygame as pg

from src.core import Vec3


class RenderResult:
    def __init__(self, width: int, height: int, render_passes: int) -> None:
        self.RENDER_PASSES = render_passes

        self.WIDTH: int = width
        self.HEIGHT: int = height
        self.PX_NUM: int = width * height
        self.arr: np.ndarray = np.zeros(shape=(width, height, 3), dtype=np.float64)

        self.rendered_passes: int = 0
        self.rendered_pixels: int = 0
        self.finished: bool = False

        self.img_arr = np.zeros_like(self.arr, dtype=np.float64)
        self.img_arr_clipped = np.zeros_like(self.arr, dtype=np.float64)
        self.surface = pg.Surface((self.WIDTH, self.HEIGHT))

    def add_to_px(self, x: int, y: int, v: Vec3) -> None:
        if 0 <= x < self.arr.shape[0] and 0 <= y < self.arr.shape[1]:
            self.arr[x, y] += v
            self.rendered_pixels += 1

    def save(self, path) -> None:
        self.update_surface()
        pg.image.save(self.surface, path)

    @property
    def progress(self) -> float:
        return self.rendered_pixels / self.PX_NUM / self.RENDER_PASSES

    @property
    def progress_percent(self) -> float:
        return self.rendered_pixels / self.PX_NUM / self.RENDER_PASSES * 100

    def update_views(self):
        self.update_img_arr()
        self.update_img_arr_clipped()

    def update_img_arr(self):
        np.copyto(dst=self.img_arr, src=self.arr[:, ::-1] / self.rendered_passes)

    def update_img_arr_clipped(self):
        np.copyto(
            dst=self.img_arr_clipped,
            src=np.clip(self.arr[:, ::-1] / self.rendered_passes, 0, 1),
        )

    def update_surface(self):
        self.surface = pg.surfarray.make_surface(self.img_arr_clipped * 255)
