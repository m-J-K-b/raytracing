import sys

import numpy as np
import pygame as pg

from scenes import scene1
from src.material import Material
from src.objects.object_sphere import Sphere
from src.post_processing import PostProcessing
from src.renderer import Renderer
from src.util import Vec3


def main():
    pg.init()
    WIDTH, HEIGHT = int(1080 * 0.5), int(1920 * 0.5)
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()

    renderer = Renderer(int(1080), int(1920))
    post_processing = PostProcessing(exposure=1.2)

    renderer.scene = scene1.get_scene()
    renderer.scene.objects.append(
        Sphere(
            Material(Vec3(1), Vec3(1), 1, 0, Vec3(1), 0, Vec3(1), 1.45),
            Vec3(500, 500, 0),
            300,
        )
    )
    renderer.scene.environment_image = None
    renderer.max_passes = 10
    renderer.max_bounces = 10

    start_cam_fov = np.pi / 1.5
    target_cam_fov = 0.1
    total_frames = 2
    current_frame = 2

    renderer.output_path = f"./renders/img{current_frame}.png"
    renderer.scene.camera.set_lookat(Vec3(0, 3, 50))
    renderer.scene.camera.dof_strength = 1

    current_cam_fov = (
        current_frame / total_frames * (target_cam_fov - start_cam_fov) + start_cam_fov
    )
    current_cam_z = 10 - 6 / np.tan(current_cam_fov / 2)
    renderer.scene.camera.pos = Vec3(0, 3, current_cam_z)
    renderer.scene.camera.set_fov(current_cam_fov)
    renderer.scene.camera.dof_dist = 50 - current_cam_z
    current_frame += 1

    t = renderer.render_threaded(5, 5)

    while True:
        if not t.is_alive():
            pg.image.save(
                pg.transform.flip(
                    pg.surfarray.array3d(
                        post_processing.process(renderer.get_img_arr())
                    ),
                    False,
                    True,
                ),
                f"./renders/img__{current_frame}.png",
            )
            current_cam_fov = (
                current_frame / (total_frames - 1) * (target_cam_fov - start_cam_fov)
                + start_cam_fov
            )
            current_cam_z = 10 - 6 / np.tan(current_cam_fov / 2)
            renderer.scene.camera.pos = Vec3(0, 3, current_cam_z)
            renderer.scene.camera.set_fov(current_cam_fov)
            renderer.scene.camera.dof_dist = 50 - current_cam_z
            current_frame += 1
            if current_frame > total_frames:
                pg.quit()
                sys.exit()

            t = renderer.render_threaded(5, 5)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        scaled_surface = pg.transform.flip(
            pg.transform.scale(
                pg.surfarray.make_surface(
                    post_processing.process(renderer.get_img_arr()) * 255
                ),
                RES,
            ),
            False,
            True,
        )
        screen.blit(scaled_surface, (0, 0))

        pg.display.set_caption(
            f"Frame: {current_frame}, pass: {renderer.rendered_passes} / {renderer.max_passes}, rendering: {renderer.rendering}"
        )

        pg.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
