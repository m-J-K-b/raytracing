import pygame as pg
from pygame import Vector2 as vec2


def main():
    pg.init()
    WIDTH, HEIGHT = 1600, 800
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)

    bounds = [[200, 200], [WIDTH - 200, HEIGHT - 200]]

    ray_origin = vec2(100, 500)
    ray_dir = vec2(1, 0)

    move_ray_origin = False
    selection_radius = 30

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        mouse_position = pg.mouse.get_pos()
        mouse_buttons = pg.mouse.get_pressed()

        move_ray_origin = False
        if (
            mouse_buttons[0]
            and ray_origin.distance_squared_to(mouse_position) < selection_radius**2
        ):
            move_ray_origin = True

        if move_ray_origin:
            ray_origin = vec2(mouse_position)
        else:
            if mouse_position[0] != ray_origin.x or mouse_position[1] != ray_origin.y:
                ray_dir = (vec2(mouse_position) - ray_origin).normalize()

        screen.fill((30, 30, 30))

        pg.draw.line(
            screen,
            (255, 255, 255),
            (bounds[0][0], bounds[0][1]),
            (bounds[0][0], bounds[1][1]),
        )
        pg.draw.line(
            screen,
            (255, 255, 255),
            (bounds[0][0], bounds[1][1]),
            (bounds[1][0], bounds[1][1]),
        )
        pg.draw.line(
            screen,
            (255, 255, 255),
            (bounds[1][0], bounds[1][1]),
            (bounds[1][0], bounds[0][1]),
        )
        pg.draw.line(
            screen,
            (255, 255, 255),
            (bounds[1][0], bounds[0][1]),
            (bounds[0][0], bounds[0][1]),
        )

        pg.draw.circle(screen, (255, 0, 0), ray_origin, 20)
        pg.draw.line(screen, (255, 255, 255), ray_origin, ray_origin + ray_dir * 1000)
        pg.draw.line(screen, (255, 255, 255), ray_origin, ray_origin - ray_dir * 1000)

        tx1, tx2 = (bounds[0][0] - ray_origin.x) / (ray_dir.x + 1e-5), (
            bounds[1][0] - ray_origin.x
        ) / (ray_dir.x + 1e-5)
        if tx1 > tx2:
            temp = tx1
            tx1 = tx2
            tx2 = temp
        ty1, ty2 = (bounds[0][1] - ray_origin.y) / (ray_dir.y + 1e-5), (
            bounds[1][1] - ray_origin.y
        ) / (ray_dir.y + 1e-5)
        if ty1 > ty2:
            temp = ty1
            ty1 = ty2
            ty2 = temp
        t1, t2 = max(tx1, ty1), min(tx2, ty2)
        overlap1 = (ty2 - ty1) - (tx1 - ty1)
        overlap2 = (tx2 - tx1) - (ty1 - tx1)
        inside = overlap1 > 0 and overlap2 > 0

        pg.display.set_caption(
            f"{ray_dir}, tx1: {tx1:.4f}, tx2: {tx2:.4f}, ty1: {ty1:.4f}, ty2: {ty2:.4f}, t1: {t1:.4f}, t2: {t2:.4f}, overlap1: {overlap1}, overlap2: {overlap2}"
        )
        # pg.draw.line(screen, (255, 128, 128), ray_origin + ray_dir * tx1, ray_origin + ray_dir * tx2, 10)
        # pg.draw.line(screen, (128, 255, 127), ray_origin + ray_dir * ty1, ray_origin + ray_dir * ty2, 4)
        if inside:
            pg.draw.line(
                screen,
                (0, 0, 255),
                ray_origin + ray_dir * t1,
                ray_origin + ray_dir * t2,
                4,
            )

        pg.display.update()


if __name__ == "__main__":
    main()
