import sys
from colorsys import hsv_to_rgb

import numpy as np
import pygame as pg

from src import *


def main():
    pg.init()
    WIDTH, HEIGHT = 1000, 1000
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()
    r = Renderer()
    s = Scene()
    pp = PostProcessing()
    pp.exposure = 0.8
    # pp.brightness = -0.3
    sphere0 = objects.Sphere(
        material=Material(
            color=Vec3(0.98, 0.88, 0.5),
            emission_strength=0,
            smoothness=0.8,
            transmittance=0,
            ior=1.45,
        ),
        origin=Vec3(0, 11, 0),
        radius=10,
    )
    sphere1 = objects.Sphere(
        material=Material(
            color=Vec3(0.5, 0.88, 0.98),
            emission_strength=0,
            smoothness=0.8,
            transmittance=0,
            ior=1.45,
        ),
        origin=Vec3(0, -11, 0),
        radius=10,
    )
    sphere2 = objects.Sphere(
        material=Material(
            color=Vec3(1),
            emission_strength=0.2,
            smoothness=0,
            transmittance=0,
            ior=1.45,
        ),
        origin=Vec3(0),
        radius=1,
    )
    sphere3 = objects.Sphere(
        material=Material(
            color=Vec3(0.7, 0.7, 1),
            emission_strength=0.8,
            smoothness=0,
            transmittance=0,
            ior=1.45,
        ),
        origin=Vec3(-200, 0, -200),
        radius=170,
    )
    sphere4 = objects.Sphere(
        material=Material(
            color=Vec3(1, 0.7, 0.7),
            emission_strength=0.8,
            smoothness=0,
            transmittance=0,
            ior=1.45,
        ),
        origin=Vec3(200, 0, -200),
        radius=170,
    )
    s.add_object(sphere0)
    s.add_object(sphere1)
    s.add_object(sphere2)
    s.add_object(sphere3)
    s.add_object(sphere4)
    # s.set_environment(pg.image.load("./assets/skyboxes/skybox1.png"))
    s.camera = Camera(Vec3(0, 0, -10), np.pi / 3, Vec3(0), dof_dist=9.5, dof_strength=1)
    rs = RenderSettings(s, 1000, 1000, 100, 8)

    t = 0
    dt = 0

    total_time = 0

    def convert_seconds(seconds: int):
        s = int(seconds)
        return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"

    rr = r.render_threaded(rs, 4, 4)
    while True:
        t += dt
        if not rr.finished:
            total_time = t / (rr.progress + 1e-9)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        scaled_img = pg.transform.scale(
            pg.surfarray.make_surface(pp.process(rr.img_arr) * 255),
            RES,
        )
        # if rr.finished:
        #     pg.image.save(rr.img, f"./renders/transmissions1.png")
        #     pg.quit()
        #     sys.exit()
        # scaled_img = pg.transform.scale(
        #     pg.surfarray.make_surface(
        #         (np.clip(rr.depth_arr, 0, np.pi / 2) / np.pi / 2)[:, :, None]
        #         * np.array([255, 255, 255])
        #     ),
        #     RES,
        # )
        screen.blit(scaled_img, (0, 0))
        pg.display.set_caption(
            f"rendering: {rr.progress_percent:.4f}%, estimated {convert_seconds(t)} / {convert_seconds(total_time)} (h:m:s)"
        )
        pg.display.update()
        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()
