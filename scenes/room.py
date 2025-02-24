import random

import numpy as np
import pygame as pg

from src.core import Material, Ray, Vec3
from src.mesh import Mesh, Sphere
from src.raytracing import Camera, Scene, util

sky_map: pg.Surface = None
SKY_MAP_W = None
SKY_MAP_H = None


def sample_sky_map(ray: Ray):
    if sky_map == None:
        return Vec3(0)
    u, v = util.vec_to_sky_coords(ray.direction)
    x, y = int(u * SKY_MAP_W), int(v * SKY_MAP_H)
    c = sky_map.get_at((x, y))
    c = Vec3(c[0], c[1], c[2]) / 255
    return c * 0.4


def get_scene():
    global sky_map, SKY_MAP_W, SKY_MAP_H
    sky_map = pg.image.load("./assets/skyboxes/skybox1.png").convert()
    SKY_MAP_W = sky_map.get_width()
    SKY_MAP_H = sky_map.get_height()

    objects = Mesh.load_from_obj_file("./assets/models/obj/room.obj")
    for o in objects:
        if o.name == "left":
            o.material = Material(Vec3(1, 0.5, 0.5), 0, 0, 0, 1)
        elif o.name == "back":
            o.material = Material(Vec3(0.5, 1, 0.5), 0, 0, 0, 1)
        elif o.name == "right":
            o.material = Material(Vec3(0.5, 0.5, 1), 0, 0, 0, 1)
        elif o.name == "lamp":
            o.material = Material(Vec3(1), 1, 0, 0, 1)
        else:
            o.material = Material(Vec3(1), 0, 0, 0, 1)
    objects.append(Sphere(Material(Vec3(1), 0, 0.3, 0.1, 1.4), Vec3(0), 0.5))

    camera = Camera(Vec3(0, 0, -2), np.pi / 2, Vec3(0))

    scene = Scene()
    scene.get_environment = sample_sky_map
    scene.objects = objects
    scene.camera = camera

    return scene
