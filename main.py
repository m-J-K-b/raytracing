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

    s = objects.Sphere(Material.default_material(), Vec3(0, -100000, 0), 100000)
    s.material.color = Vec3(1)
    s.material.smoothness = 0
    s.material.emission_strength = 0
    s.material.transmittance = 0
    scene.add_object(s)

    s = objects.Sphere(Material.default_material(), Vec3(5000), 5000)
    s.material.color = Vec3(1)
    s.material.smoothness = 0
    s.material.emission_strength = 1
    s.material.transmittance = 0
    scene.add_object(s)

    positions = []
    radii = []
    while len(positions) < 200:
        p = Vec3.random_unit().elementwise() * Vec3(1, 0, 1) * 100
        r = 1 / (np.random.random() + 0.95) ** 17 + 1.5
        valid = True
        for p1, r1 in zip(positions, radii):
            if (p1 - p).magnitude() < r + r1:
                valid = False
                break
        if not valid:
            continue  # Skip adding this sphere and try again.
        positions.append(p)
        radii.append(r)

    for p, r in zip(positions, radii):
        p.y += r
        s = objects.Sphere(Material.default_material(), p, r)

        v = np.random.random()
        if v < 1 / 3:
            print("metal")
            s.material.color = Vec3.random_unit().absolute()
            s.material.smoothness = np.random.random() * 0.8 + 0.1
            s.material.emission_strength = 0
            s.material.transmittance = 0
        elif v < 2 / 3:
            print("emissive")
            s.material.color = Vec3.random_unit().absolute()
            s.material.smoothness = 0
            s.material.emission_strength = np.random.random() + 0.5
            s.material.transmittance = 0
        else:
            print("glass")
            s.material.color = Vec3(1)
            s.material.smoothness = 0
            s.material.emission_strength = 0
            s.material.transmittance = np.random.random() * 0.5 + 0.5

        scene.add_object(s)

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
