from typing import List

import numpy as np
import pygame as pg

from src.util import Vec3


class RenderResult:
    def __init__(self, width: int, height: int, render_passes: int) -> None:
        self.RENDER_PASSES = render_passes

        self.WDITH: int = width
        self.HEIGHT: int = height
        self.PX_NUM: int = width * height
        self.arr: List[float] = np.zeros(shape=(width, height, 3), dtype=np.float64)

        self.rendered_passes: int = 0
        self.rendered_pixels: int = 0
        self.finished: bool = False

    def add_to_px(self, x: int, y: int, v: Vec3) -> None:
        if 0 <= x < self.arr.shape[0] and 0 <= y < self.arr.shape[1]:
            self.arr[x, y] += v
            self.rendered_pixels += 1

    def save(self, path) -> None:
        pg.image.save(self.img, path)

    @property
    def progress(self) -> float:
        return self.rendered_pixels / self.PX_NUM / self.RENDER_PASSES

    @property
    def progress_percent(self) -> float:
        return self.rendered_pixels / self.PX_NUM / self.RENDER_PASSES * 100

    @property
    def img_arr(self):
        return self.arr[:, ::-1] / self.rendered_passes

    @property
    def img_arr_clipped(self):
        return np.clip(self.arr[:, ::-1] / self.rendered_passes, 0, 1)

    @property
    def img(self):
        return pg.surfarray.make_surface(self.img_arr_clipped * 255)
