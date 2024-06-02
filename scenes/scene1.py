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
    scene = Scene()
    random.seed(6)

    materials = []
    for i in range(50):
        mat = Material.default_material()
        if random.random() < 0.2:
            mat.emission_color = Vec3(random.random(), random.random(), random.random())
            mat.emission_strength = 1
            mat.color = Vec3(1)
            mat.smoothness = 0
            mat.specular_color = Vec3(1)
            mat.specular_probability = 0
        else:
            mat.color = Vec3(random.random(), random.random(), random.random())
            mat.specular_color = Vec3(random.random(), random.random(), random.random())
            mat.specular_probability = random.random()
            mat.smoothness = random.random()
            mat.emission_strength = 0
            mat.emission_color = Vec3(random.random(), random.random(), random.random())
        materials.append(mat)

    big_sphere_mat = Material.default_material()
    big_sphere_mat.color = Vec3(252, 172, 68) / 255
    big_sphere_mat.emission_strength = 1
    big_sphere_mat.emission_color = Vec3(252, 172, 60) / 255
    big_sphere_mat.smoothness = 0
    big_sphere_mat.specular_probability = 0
    big_sphere_mat.specular_color = Vec3(1)

    floor_mat = Material.default_material()
    floor_mat.color = Vec3(0.8)
    floor_mat.emission_color = Vec3(1)
    floor_mat.specular_color = Vec3(1)
    floor_mat.emission_strength = 0
    floor_mat.smoothness = 0
    floor_mat.specular_probability = 0

    horse_mat = Material.default_material()
    horse_mat.color = Vec3(0.7, 0.7, 0.8)
    horse_mat.emission_color = Vec3(1)
    horse_mat.specular_color = Vec3(1)
    horse_mat.emission_strength = 0
    horse_mat.smoothness = 0.3
    horse_mat.specular_probability = 0

    horse = Mesh.load_from_obj_file("./assets/models/obj/chess_horse.obj")[0]
    horse.material = horse_mat
    horse.set_origin(Vec3(0, 0, 50))
    horse.set_scale(Vec3(3, 3, 3))

    objects = [
        SimplePlane(floor_mat, 0),
        Sphere(big_sphere_mat, Vec3(0, 100, 1000), 100),
        horse,
    ] + [
        Sphere(
            random.choice(materials),
            Vec3(6 * (1 if i % 2 == 0 else -1), 3, (i // 2) * 20 + 10),
            3,
        )
        for i in range(0, 10)
    ]
    camera = Camera(
        Vec3(0, 3, 0), np.pi / 10, Vec3(0, 100, 1000), dof_strength=0.1, dof_dist=50
    )

    for obj in objects:
        scene.add_object(obj)

    scene.camera = camera
    scene.set_environment(pg.image.load("./assets/skyboxes/skybox1.png"))

    return scene
