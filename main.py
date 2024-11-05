import math
import sys
import time
from colorsys import hsv_to_rgb
from datetime import datetime
from threading import Thread

import numpy as np
import pygame as pg

from sky_shader import get_sky_color, worley_noise
from src import *


def convert_seconds(seconds: int | float):
    s = int(seconds)
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def random_offset(u, v):
    # Generate a pseudo-random offset based on u and v
    seed = int(u * 1000) + int(v * 1000)  # Create a unique seed based on coordinates
    np.random.seed(seed)  # Seed the random number generator
    return np.random.uniform(-0.1, 0.1)  # Random offset in the range [-0.1, 0.1]


def get_env_color_bands(ray: Ray) -> Vec3:
    u, v = util.vec_to_sky_coords(ray.direction)
    if math.isnan(u) or math.isnan(v):
        return Vec3(1)

    N = 100
    offset = random_offset(u, v)  # Use the simple random offset
    # u += offset * 0.1
    u = int((u + offset * 0.1) * N % N) / N

    return Vec3(
        np.sin(u * np.pi * 50) * 0.5 + 0.5,
        np.sin(v * np.pi * 4 + u * 40) * 0.5 + 0.5,
        0.5,
    )


def get_env_simple_sky(ray: Ray) -> Vec3:
    return Vec3(*get_sky_color(ray.direction, 10000))


def continuous_post_processing(
    input_img_arr: np.ndarray,
    output_img_arr: np.ndarray,
    post_processing: PostProcessing,
    **kwargs,
) -> Thread:

    def _continuous_post_processing(
        input_img_arr: np.ndarray,
        output_img_arr: np.ndarray,
        post_processing: PostProcessing,
        **kwargs,
    ) -> None:
        callback = kwargs.get("callback")
        while True:
            np.copyto(dst=output_img_arr, src=post_processing.process(input_img_arr))
            if callable(callback):
                callback()

    t = Thread(
        target=_continuous_post_processing,
        args=(
            input_img_arr,
            output_img_arr,
            post_processing,
        ),
        kwargs=kwargs,
        daemon=True,
    )
    t.start()
    return t


def main():
    pg.init()
    WIDTH, HEIGHT = int(600 * 1.4), int(800 * 1.4)
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()
    t = 0
    dt = 0
    total_time = 0

    scene = Scene()

    scene.get_environment = get_env_color_bands

    horse = objects.Mesh.load_from_obj_file("./assets/models/obj/chess_horse.obj")[0]
    horse.set_rotation(Vec3(0, 180, 0))
    horse.set_scale(Vec3(1.3))
    horse.set_origin_to_center_of_mass()
    horse.set_origin(Vec3(0, 0, 0))
    horse.material.smoothness = 0.7
    horse.material.color = Vec3(1)
    scene.add_object(horse)

    obj = objects.Sphere(Material.default_material(), Vec3(0, -7, 0), 5)
    obj.material.color = Vec3(0.8, 0.8, 1)
    scene.add_object(obj)

    scene.camera = Camera(
        Vec3(-7, -1.1, 0),
        np.pi / 4,
        Vec3(0, -1, 0),
        dof_strength=0.05,
        dof_dist=(Vec3(-7, -0.1, 0) - horse.origin).magnitude(),
    )
    # T = np.pi / 3 * 4
    # scene.camera = Camera(
    #     Vec3(0),
    #     np.pi / 1.5,
    #     Vec3(np.cos(T), 0.2, np.sin(T)),
    #     dof_strength=0,
    #     dof_dist=0,
    # )
    # scene.camera.pos.x = 5 / np.sin(scene.camera.fov / 2)
    # scene.camera.dof_dist = abs(scene.camera.pos.x)

    render_settings = RenderSettings(scene, int(600 * 2), int(800 * 2), 15, 4)
    renderer = Renderer()
    post_processing = PostProcessing()
    post_processing.exposure = 1
    post_processing.brightness = 0
    post_processing.contrast = 1.2
    post_processing.saturation = 0.5
    post_processing.gamma

    render_result = renderer.start_render_threaded(render_settings, 3, 4)
    processed_result = np.zeros_like(render_result.img_arr)

    def callback(*args, **kwargs):
        nonlocal render_result
        render_result.update_views()

    post_processing_thread = continuous_post_processing(
        render_result.img_arr, processed_result, post_processing, callback=callback
    )

    saved = False
    while True:
        t += dt
        if not render_result.finished:
            total_time = t / (render_result.progress + 1e-9)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
        if render_result.finished and not saved:
            render_result.update_views()
            render_result.img_arr.tofile(
                f"image_{render_result.WIDTH}x{render_result.HEIGHT}_{render_result.img_arr.dtype}.raw"
            )

            pg.image.save(
                pg.surfarray.make_surface(
                    post_processing.process(render_result.img_arr) * 255
                ),
                f"./renders/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.png",
            )

            from PIL import Image

            img = Image.fromarray(render_result.img_arr.astype(np.uint8))
            img.save("full_dynamic_range.tiff", format="TIFF")
            saved = True

        img = pg.surfarray.make_surface(processed_result * 255)
        scaled_img = pg.transform.scale(
            img,
            RES,
        )
        screen.blit(scaled_img, (0, 0))

        pg.display.set_caption(
            f"rendering: {render_result.progress_percent:.4f}%, estimated {convert_seconds(t)} / {convert_seconds(total_time)}"
        )

        pg.display.update()
        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()
