import sys

import pygame as pg

from scenes import scene1
from src.renderer import Renderer


def main():
    pg.init()
    WIDTH, HEIGHT = 500, 500
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()

    renderer = Renderer(int(WIDTH / 5), int(HEIGHT / 5))

    renderer.scene = scene1.get_scene()
    renderer.max_bounces = 10
    renderer.render_threaded(5, 5)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    renderer.break_[0] = not renderer.break_[0]

        scaled_surface = pg.transform.flip(
            pg.transform.scale(renderer.get_img_post_processed(), RES), False, True
        )
        screen.blit(scaled_surface, (0, 0))

        pg.display.set_caption(f"Iteration: {renderer.passes[0]}, {renderer.break_[0]}")

        pg.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
