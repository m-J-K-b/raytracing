import math
import sys
import time
from datetime import datetime
from threading import Thread

import numpy as np
import pygame as pg

from src import *
from src.bxdf import DiffuseBRDF


def convert_seconds(seconds: int | float):
    s = int(seconds)
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def random_offset(u, v):
    # Generate a pseudo-random offset based on u and v
    return np.random.uniform(-0.1, 0.1)  # Random offset in the range [-0.1, 0.1]


def get_env_color_bands(ray: Ray) -> Vec3:
    # return Vec3(0)
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

    obj = objects.Sphere(Material.default_material(), DiffuseBRDF(), Vec3(0), 1)
    obj.material.color = Vec3(1, 0.8, 0.5)
    scene.add_object(obj)

    obj = objects.Sphere(
        Material.default_material(), DiffuseBRDF(), Vec3(0, -100, 0), 99
    )
    obj.material.color = Vec3(1)
    scene.add_object(obj)

    obj = objects.Sphere(Material.default_material(), DiffuseBRDF(), Vec3(0, 10, 10), 5)
    obj.material.color = Vec3(1)
    obj.material.emission_strength = 1
    scene.add_object(obj)

    scene.camera = Camera(
        Vec3(5, 5, -4),
        90 * np.pi / 180,
        Vec3(0, 3, 5),
        dof_strength=0,
    )

    render_settings = RenderSettings(scene, int(600), int(800), 15, 4)
    renderer = Renderer()
    post_processing = PostProcessing()
    post_processing.exposure = 1
    post_processing.brightness = 0
    post_processing.contrast = 1.2
    post_processing.saturation = 0.5
    post_processing.gamma

    render_result = renderer.start_render_threaded(render_settings, 3, 4)
    # render_result = renderer.start_render(render_settings)
    processed_result = np.zeros_like(render_result.img_arr)
    img = pg.surfarray.make_surface(processed_result * 255)

    def callback(*args, **kwargs):
        nonlocal render_result, img
        render_result.update_views()
        img = pg.surfarray.make_surface(processed_result * 255)

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
