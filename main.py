import math
import sys
from colorsys import hsv_to_rgb

import numpy as np
import pygame as pg

from src import *

def convert_seconds(seconds: int):
    s = int(seconds)
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def get_env(ray: Ray):
    u, v = util.vec_to_sky_coords(ray.direction)
    if math.isnan(u) or math.isnan(v):
        return Vec3(1)
    values = [
        (0, Vec3(0)),
        (0.2, Vec3(0)),
        (0.25, Vec3(1, 1, 0.5)),
        (0.4, Vec3(0.3)),
        (0.6, Vec3(0.3)),
        (0.75, Vec3(0.5, 1, 1)),
        (0.8, Vec3(0)),
        (1, Vec3(0)),
    ]
    d = min(1, ((u % 0.5 - 0.25) ** 2 + (v - 0.5) ** 2) ** 0.5 * 10)
    # return Vec3(u, v, 0.5)
    for i in range(len(values)):
        t, c = values[i]
        next_t, next_c = values[i + 1]
        if t < u < next_t:
            local_t = (u - t) / (next_t - t)
            lerped_c = util.lerp(c, next_c, local_t)
            # return util.lerp(lerped_c, Vec3(0), d)
            return lerped_c


def main():
    pg.init()
    WIDTH, HEIGHT = int(1080 / 1.5), int(1920 / 1.5)
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()
    t = 0
    dt = 0
    total_time = 0

    renderer = Renderer()
    scene = Scene()
    post_processing = PostProcessing()
    post_processing.exposure = 2
    post_processing.saturation = 0.8
    # pp.brightness = -0.3
    obj = objects.Mesh.load_from_obj_file("./assets/models/obj/chess_horse.obj")[0]
    
    scene.add_object(obj)
    scene.get_environment = get_env
    # scene.set_environment(pg.image.load("./assets/skyboxes/skybox1.png"))
    scene.camera = Camera(
        Vec3(5, 0, 0), np.pi / 4, Vec3(0), dof_dist=10, dof_strength=2
    )
    scene.camera.pos.x = 5 / np.sin(scene.camera.fov / 2)
    scene.camera.dof_dist = abs(scene.camera.pos.x)
    render_settings = RenderSettings(scene, int(1080 * 0.4), int(1920 * 0.4), 100, 10)

    render_result = renderer.start_render_threaded(render_settings, 4, 4)
    while True:
        t += dt
        if not render_result.finished:
            total_time = t / (render_result.progress + 1e-9)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        if render_result.finished:
            pg.image.save(
                pg.surfarray.make_surface(
                    post_processing.process(render_result.img_arr) * 255
                ),
                f"./renders/big_spehr123er.png",
            )
        #     n += 1
        #     if n == N:
        #         quit()
        #         sys.exit()
        #     scene.camera.pos = cam_pos()
        #     scene.camera.update_axis()
        #     render_result = renderer.render(render_settings)

        img = pg.surfarray.make_surface(
            post_processing.process(render_result.img_arr) * 255
        )

        scaled_img = pg.transform.scale(
            img,
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
            f"rendering: {render_result.progress_percent:.4f}%, estimated {convert_seconds(t)} / {convert_seconds(total_time)} (h:m:s)"
        )
        pg.display.update()
        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()