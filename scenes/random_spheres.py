import random

import numpy as np
import pygame as pg

from src import core, mesh, raytracing

sky_map: pg.Surface = None
SKY_MAP_W = None
SKY_MAP_H = None


def sample_sky_map(ray: core.Ray):
    if sky_map == None:
        return Vec3(0)
    u, v = raytracing.util.vec_to_sky_coords(ray.direction)
    x, y = int(u * SKY_MAP_W), int(v * SKY_MAP_H)
    c = sky_map.get_at((x, y))
    c = core.Vec3(c[0], c[1], c[2]) / 255
    return c


def get_scene():
    global sky_map, SKY_MAP_W, SKY_MAP_H
    sky_map = pg.image.load("./assets/skyboxes/skybox1.png").convert()
    SKY_MAP_W = sky_map.get_width()
    SKY_MAP_H = sky_map.get_height()

    scene = raytracing.Scene()

    scene.get_environment = sample_sky_map

    s = mesh.Sphere(core.Material.default_material(), core.Vec3(0, -100000, 0), 100000)
    s.material.color = core.Vec3(1)
    s.material.smoothness = 0
    s.material.emission_strength = 0
    s.material.transmittance = 0
    scene.add_object(s)

    s = mesh.Sphere(core.Material.default_material(), core.Vec3(5000), 5000)
    s.material.color = core.Vec3(1)
    s.material.smoothness = 0
    s.material.emission_strength = 1
    s.material.transmittance = 0
    scene.add_object(s)

    positions = []
    radii = []
    while len(positions) < 200:
        p = core.Vec3.random_unit().elementwise() * core.Vec3(1, 0, 1) * 100
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
        s = mesh.Sphere(core.Material.default_material(), p, r)

        v = np.random.random()
        if v < 1 / 3:
            print("metal")
            s.material.color = core.Vec3.random_unit().absolute()
            s.material.smoothness = np.random.random() * 0.8 + 0.1
            s.material.emission_strength = 0
            s.material.transmittance = 0
        elif v < 2 / 3:
            print("emissive")
            s.material.color = core.Vec3.random_unit().absolute()
            s.material.smoothness = 0
            s.material.emission_strength = np.random.random() + 0.5
            s.material.transmittance = 0
        else:
            print("glass")
            s.material.color = core.Vec3(1)
            s.material.smoothness = 0
            s.material.emission_strength = 0
            s.material.transmittance = np.random.random() * 0.5 + 0.5

        scene.add_object(s)

    scene.camera = raytracing.Camera(
        core.Vec3(0, 0, 5),  # Camera position
        np.pi / 2,  # Field of view
        core.Vec3(0, 0, 0),  # Look-at point
        dof_strength=0.01,
        dof_dist=10,
    )

    return scene
