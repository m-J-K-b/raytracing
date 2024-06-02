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
    np.random.seed(1)
    objects = []

    plane = SimplePlane(Material.random(), 0)
    objects.append(plane)

    for i in range(400):
        x, z = random.random() * 200 - 100, random.random() * 200 - 100
        y = random.random() * 5 + 0.2
        objects.append(Sphere(Material.random(), Vec3(x, y, z), y))
        objects[-1].material.emission_strength = 1 if np.random.random() < 0.1 else 0

    scene = Scene()
    scene.objects = objects
    scene.camera = Camera(Vec3(-100, 4, 0), np.pi / 2, Vec3(0, 0, 0), dof_strength=0.5)
    scene.camera.dof_dist = (scene.camera.pos - scene.camera.lookat).magnitude()
    scene.environment_image = pg.image.load("./assets/skyboxes/skybox1.png").convert()

    return scene
