import math
import sys
import time
from colorsys import hsv_to_rgb
from datetime import datetime
from threading import Thread

import numpy as np
import pygame as pg
from PIL import Image

from src import *
from src.util import vec_to_sky_coords


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
        np.sin(u * np.pi * 1) * 0.5 + 0.5,
        np.sin(v * np.pi * 4 + u * 40) * 0.5 + 0.5,
        0.5,
    )


sky_map: pg.Surface = None
SKY_MAP_W = None
SKY_MAP_H = None


def sample_sky_map(ray: Ray):
    if sky_map == None:
        return Vec3(0)
    u, v = vec_to_sky_coords(ray.direction)
    x, y = int(u * SKY_MAP_W), int(v * SKY_MAP_H)
    c = sky_map.get_at((x, y))
    c = Vec3(c[0], c[1], c[2]) / 255
    return c


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
    WIDTH, HEIGHT = int(800), int(400)
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()
    t = 0
    dt = 0
    total_time = 0

    global sky_map, SKY_MAP_W, SKY_MAP_H
    sky_map = pg.image.load("./assets/skyboxes/skybox1.png").convert()
    SKY_MAP_W = sky_map.get_width()
    SKY_MAP_H = sky_map.get_height()

    scene = Scene()

    scene.get_environment = sample_sky_map

    # Sphere 1: Reflective Red Sphere
    sphere1 = objects.Sphere(Material.default_material(), Vec3(-1, 0, 0), 1)
    sphere1.material.color = Vec3(1, 0, 0)  # Red
    sphere1.material.transmittance = 0.0
    sphere1.material.smoothness = 0.9
    sphere1.material.ior = 1.5  # Reflective
    scene.add_object(sphere1)

    # Sphere 2: Transparent Blue Sphere
    sphere2 = objects.Sphere(Material.default_material(), Vec3(-3, 0, 0), 1)
    sphere2.material.color = Vec3(0, 0, 1)  # Blue
    sphere2.material.transmittance = 0.8  # Transparent
    sphere2.material.smoothness = 0.1
    sphere2.material.ior = 1.33  # Glass-like
    scene.add_object(sphere2)

    # Sphere 3: Matte Green Sphere
    sphere3 = objects.Sphere(Material.default_material(), Vec3(1, 0, 0), 1)
    sphere3.material.color = Vec3(0, 1, 0)  # Green
    sphere3.material.transmittance = 0.0
    sphere3.material.smoothness = 0.0  # Matte
    sphere3.material.ior = 1.0
    scene.add_object(sphere3)

    # Sphere 4: Emissive Yellow Sphere
    sphere4 = objects.Sphere(Material.default_material(), Vec3(3, 0, 0), 1)
    sphere4.material.color = Vec3(1, 1, 0)  # Yellow
    sphere4.material.transmittance = 0.0
    sphere4.material.smoothness = 0.0
    sphere4.material.ior = 1.0
    sphere4.material.emission_strength = 1.0  # Emits light
    scene.add_object(sphere4)

    sphere5 = objects.Sphere(Material.default_material(), Vec3(0, -101, 0), 100)
    sphere5.material.color = Vec3(1, 1, 1)  # Yellow
    sphere5.material.transmittance = 0.0
    sphere5.material.smoothness = 0.0
    sphere5.material.ior = 1.0
    sphere5.material.emission_strength = 0.0  # Emits light
    scene.add_object(sphere5)

    scene.camera = Camera(
        Vec3(0, 0, 5),  # Camera position
        np.pi / 2,  # Field of view
        Vec3(0, 0, 0),  # Look-at point
        dof_strength=0.01,
        dof_dist=10,
    )

    render_settings = RenderSettings(scene, int(800), int(400), 30, 4)
    renderer = Renderer()
    post_processing = PostProcessing()

    render_result = renderer.start_render(render_settings, 3, 4)
    # render_result = renderer.start_render(render_settings)
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
                renderer.stop_render()
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
