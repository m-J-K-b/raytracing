import sys
from datetime import datetime
from threading import Thread

import numpy as np
import pygame as pg

from src import core, mesh, post_processing, raytracing
from src.core.material import Material
from src.core.vec3 import Vec3


def convert_seconds(seconds: int | float):
    s = int(seconds)
    return f"{s // 3600 % 24:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


sky_map: pg.Surface = None
SKY_MAP_W = None
SKY_MAP_H = None


def sample_sky_map(ray: core.Ray):
    if sky_map == None:
        return Vec3(0)
    u, v = raytracing.util.vec_to_sky_coords(ray.direction)
    x, y = int(u * SKY_MAP_W), int(v * SKY_MAP_H)
    c = sky_map.get_at((x, y))
    c = Vec3(c[0], c[1], c[2]) / 255
    return c


def main():
    pg.init()

    # Create scene and set up parameters
    # scene = raytracing.Scene()
    # sphere_count = 10
    # sphere_radius = 1.0
    # spacing_factor = 2.5
    # margin = 1.2
    # fov = np.pi / 5  # 90-degree FOV

    # Scene dimensions and resolution
    # scene_width = sphere_count * sphere_radius * spacing_factor
    # scene_height = sphere_count * sphere_radius * spacing_factor
    # resolution = (1000, 1000)

    # ------------------------------------------------------------------------
    # Populate Scene with Objects
    # ------------------------------------------------------------------------
    # for i in range(sphere_count):
    #     for j in range(sphere_count):
    #         x = (i - (sphere_count - 1) / 2) * spacing_factor * sphere_radius
    #         y = (j - (sphere_count - 1) / 2) * spacing_factor * sphere_radius

    #         v = np.random.random()
    #         mat = core.Material.default_material()
    #         if v < 1 / 3:  # emissive
    #             mat.emission_strength = 2 * np.random.random()
    #             mat.color = Vec3.random_unit().absolute()
    #         elif v < 2 / 3:  # metallic
    #             mat.color = Vec3.random_unit().absolute()
    #             mat.smoothness = np.random.random()
    #             mat.transmittance = 0
    #         else:
    #             mat.color = Vec3.random_unit().absolute()
    #             mat.smoothness = np.random.random()
    #             mat.transmittance = np.random.random()
    #             mat.ior = 1.4

    #         sphere = mesh.Sphere(mat, Vec3(x, y, 0), sphere_radius)
    #         scene.add_object(sphere)

    # ------------------------------------------------------------------------
    # Set up Camera and Environment
    # ------------------------------------------------------------------------
    # distance = (scene_width / (2 * np.tan(fov / 2))) * margin
    # scene.camera = raytracing.Camera(Vec3(0, 0, -distance), fov, Vec3(0))
    # scene.get_environment = sample_sky_map

    from scenes import room

    # ------------------------------------------------------------------------
    # Pygame Screen Setup and Time Tracking
    # ------------------------------------------------------------------------
    WIDTH, HEIGHT = (1600, 400)
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    clock = pg.time.Clock()
    t = 0
    dt = 0
    total_time = 0

    # scene = room.get_scene()
    scene = raytracing.Scene()

    # scene.get_environment = sample_sky_map

    s = mesh.Sphere(Material.default_material(), Vec3(0, -100000, 0), 100000)
    s.material.color = Vec3(1)
    s.material.smoothness = 0
    s.material.emission_strength = 0
    s.material.transmittance = 0
    scene.add_object(s)

    s = mesh.Sphere(Material.default_material(), Vec3(5000), 5000)
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
        s = mesh.Sphere(Material.default_material(), p, r)

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

    scene.camera = raytracing.Camera(
        Vec3(0, 15, 100),  # Camera position
        np.pi / 2,  # Field of view
        Vec3(0, 0, 30),  # Look-at point
        dof_strength=0.01,
        dof_dist=10,
    )

    # ------------------------------------------------------------------------
    # Render Settings and Renderer Setup
    # ------------------------------------------------------------------------
    render_settings = raytracing.RenderSettings(scene, 800, 200, 100, 4)
    renderer = raytracing.Renderer()
    post_processor = post_processing.PostProcessing()
    post_processor.exposure = 2
    post_processor.brightness = 0.02
    post_processor.contrast = 1.1
    render_result = renderer.start_render(render_settings, 16, 4)
    post_processed_img_arr = render_result.img_arr.copy()

    # # Load skybox image and related globals
    # global sky_map, SKY_MAP_W, SKY_MAP_H
    # sky_map = pg.image.load("./assets/skyboxes/skybox1.png").convert()
    # SKY_MAP_W = sky_map.get_width()
    # SKY_MAP_H = sky_map.get_height()

    # ------------------------------------------------------------------------
    # Post-Processing Callback Function
    # ------------------------------------------------------------------------
    def update_post_processing():
        nonlocal post_processed_img_arr
        while not renderer.stop_event.is_set():
            render_result.update_views()
            post_processed_img_arr = post_processor.process(render_result.img_arr)

    # Start post-processing thread
    pp_thread = Thread(target=update_post_processing, daemon=True)
    pp_thread.start()

    # ------------------------------------------------------------------------
    # Main Loop: Event Handling, Rendering Updates, and Display
    # ------------------------------------------------------------------------
    saved = False
    while True:
        t += dt
        if not render_result.finished:
            total_time = t / (render_result.progress + 1e-9)

        # Handle Pygame events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                renderer.stop_render()
                pg.quit()
                sys.exit()

        # Save the render once finished
        if render_result.finished and not saved:
            render_result.update_views()
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".png"
            img_surface = pg.surfarray.make_surface(
                post_processor.process(render_result.img_arr) * 255
            )
            pg.image.save(img_surface, f"./renders/{filename}")
            saved = True

        # Update display with post-processed image
        img = pg.surfarray.make_surface(post_processed_img_arr * 255)
        scaled_img = pg.transform.scale(img, (WIDTH, HEIGHT))
        screen.blit(scaled_img, (0, 0))

        # Update window caption with progress information
        pg.display.set_caption(
            f"rendering: {render_result.progress_percent:.4f}%, "
            f"estimated {convert_seconds(t)} / {convert_seconds(total_time)}"
        )

        pg.display.update()
        dt = clock.tick(60) / 1000


if __name__ == "__main__":
    main()
