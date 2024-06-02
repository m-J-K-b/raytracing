import random

import numpy as np
import pygame as pg

from src.camera import Camera
from src.material import Material
from src.objects.mesh import Mesh
from src.objects.simple_plane import SimplePlane
from src.objects.sphere import Sphere
from src.scene import Scene
from src.util import Vec3


def get_scene():
    objects = Mesh.load_from_obj_file("./assets/models/obj/room.obj")
    for o in objects:
        if o.name == "left":
            o.material = Material(
                Vec3(1, 0.5, 0.5), Vec3(1), Vec3(1), Vec3(1), 0, 0, 0, 1
            )
        elif o.name == "back":
            o.material = Material(
                Vec3(0.5, 1, 0.5), Vec3(1), Vec3(1), Vec3(1), 0, 0, 0, 1
            )
        elif o.name == "right":
            o.material = Material(
                Vec3(0.5, 0.5, 1), Vec3(1), Vec3(1), Vec3(1), 0, 0, 0, 1
            )
        elif o.name == "lamp":
            o.material = Material(Vec3(1), Vec3(1), Vec3(1), Vec3(1), 0, 1, 0, 1.45)
        else:
            o.material = Material(Vec3(1), Vec3(1), Vec3(1), Vec3(1), 0, 0, 0, 1.45)
    objects.append(
        Sphere(
            Material(Vec3(0.7), Vec3(1), Vec3(1), Vec3(1), 0, 0, 0.5, 1), Vec3(0), 0.5
        )
    )

    camera = Camera(Vec3(0, 0, -2), np.pi / 2, Vec3(0))

    scene = Scene()
    scene.objects = objects
    scene.camera = camera

    return scene
