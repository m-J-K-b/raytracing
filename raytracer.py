import math
import random
import sys
from copy import copy
from threading import Thread
import multiprocessing as mp

import numpy as np
import pygame as pg


def random_hemisphere_sample(normal):
    x, y, z = np.random.randn(3)
    nd = Vec3(x, y, z)
    if nd.dot(normal) < 0:
        return -nd
    return nd


def lerp(v1, v2, t):
    return v1 + (v2 - v1) * t


def quadratic_formula(a, b, c):
    return (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a), (
        -b - (b**2 - 4 * a * c) ** 0.5
    ) / (2 * a)


def smoothstep(v, minv, maxv):
    if v < minv:
        return 0
    elif v > maxv:
        return 1

    v = (v - minv) / (maxv - minv)

    return v * v * (3 - 2 * v)


class Vec3(pg.Vector3):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    def prod(self, other):
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)


class Camera:
    def __init__(self, pos, fov, lookat, dof_strength=0, dof_dist=1):
        self.pos = pos
        self.fov = fov
        self.forward = (lookat - self.pos).normalize()
        self.right = Vec3(0, 1, 0).cross(self.forward)
        self.up = self.forward.cross(self.right)
        self.d = 1 / np.tan(fov / 2)

        self.dof_strength = dof_strength
        self.dof_dist = dof_dist

    def get_ray(self, u, v):
        ray_dir = (u * self.right + v * self.up + self.forward * self.d).normalize()
        dof_target = ray_dir * self.dof_dist + self.pos
        ray_pos = (
            self.pos
            + self.right * ((random.random() * 2 - 1) * self.dof_strength)
            + self.up * ((random.random() * 2 - 1) * self.dof_strength)
        )
        ray_dir = (dof_target - ray_pos).normalize()
        return Ray(ray_pos, ray_dir)


class Material:
    def __init__(
        self,
        color,
        emission_color,
        emission_strength,
        smoothness,
        specular_color,
        specular_probability,
    ):
        self.color = color
        self.emission_color = emission_color
        self.emission_strength = emission_strength
        self.smoothness = smoothness
        self.specular_color = specular_color
        self.specular_probability = specular_probability

    @classmethod
    def default_material(self):
        return Material(Vec3(0.7), Vec3(1), 0, 0, Vec3(1), 0)

    def copy(self):
        return copy(self)


class Object:
    OBJECT_NUM = 0
    object_names = {}

    def __init__(self, material, name=None):
        self.material = material
        self.name = name
        if name in self.object_names:
            self.object_names[name] += 1
            self.name = name + self.object_names[name]

    def intersect(self, ray):
        raise NotImplementedError()


class Mesh(Object):
    def __init__(self, material, points, faces, name=None):
        super().__init__(material, name)
        self.points = points
        self.tris = faces

        self.origin = Vec3(0)
        self.scale = Vec3(1)
        self.rotation = Vec3(0)

        self.transformed_points = self.get_transformed_vertices()

    def get_scaled_vertices(self, points):
        return [p.prod(self.scale) for p in points]

    def get_translated_vertices(self, points):
        return [p + self.origin for p in points]

    def get_rotated_vertices(self):
        return self.points

    def get_transformed_vertices(self):
        vertices = self.get_scaled_vertices(self.points)
        # vertices = self.get_rotated_vertices(vertices)
        vertices = self.get_translated_vertices(vertices)
        return vertices

    def update_transformed_vertices(self):
        self.transformed_points = self.get_transformed_vertices()

    def set_scale(self, scale):
        self.scale = scale
        self.update_transformed_vertices()

    def set_origin(self, origin):
        self.origin = origin
        self.update_transformed_vertices()

    def set_rotation(self, rotation):
        self.update_transformed_vertices()

    def intersect(self, ray):
        intersections = []
        for face in self.tris:
            info = self.intersect_tri(face, ray)
            if info.hit:
                intersections.append(info)
        if intersections:
            return sorted(intersections, key=lambda x: x.hit_distance)[0]
        return HitInfo(hit=False)

    def intersect_tri(self, tri, ray):
        point_a = self.transformed_points[tri[0]]
        point_b = self.transformed_points[tri[1]]
        point_c = self.transformed_points[tri[2]]

        edge_ab = point_b - point_a
        edge_ac = point_c - point_a
        normal = edge_ab.cross(edge_ac)
        ao = ray.origin - point_a
        dao = ao.cross(ray.direction)

        determinant = -ray.direction.dot(normal)
        inv_det = 1 / determinant

        dst = normal.dot(ao) * inv_det
        u = edge_ac.dot(dao) * inv_det
        v = -edge_ab.dot(dao) * inv_det
        w = 1 - u - v

        info = HitInfo()
        if determinant >= 1e-6 and dst >= 0 and u >= 0 and v >= 0 and w >= 0:
            info.hit = True
            info.hit_distance = dst
            info.hit_pos = ray.at(dst)
            info.hit_normal = normal.normalize()
            info.hit_obj = self
        else:
            info.hit = False
        return info

    @classmethod
    def load_from_obj_file(self, path):
        with open(path, "r") as f:
            points = []
            tris = []
            vertex_index_offset = 0
            meshes = []
            for line in f.readlines():
                if line.startswith("o "):
                    if meshes:
                        meshes[-1].points = points
                        meshes[-1].tris = tris
                        vertex_index_offset += len(points)
                        points = []
                        tris = []
                    name = line.split(" ")[-1]
                    material = Material.default_material()
                    meshes.append(Mesh(material, [], [], name=name))
                elif line.startswith("v "):
                    coords = line.split(" ")[1:]
                    coords = [float(v) for v in coords]
                    points.append(Vec3(*coords))
                elif line.startswith("f "):
                    indecies = line.split(" ")[1:]
                    indecies = [
                        int(index.split("/")[0]) - 1 - vertex_index_offset
                        for index in indecies
                    ]
                    tris.append(indecies)
            else:
                meshes[-1].points = points
                meshes[-1].tris = tris
        return meshes


class Sphere(Object):
    def __init__(self, material, pos, radius, name=None):
        super().__init__(material, name)
        self.pos = pos
        self.radius = radius

    def intersect(self, ray):
        hit_info = HitInfo()
        ro = ray.origin - self.pos
        a = ray.direction.dot(ray.direction)
        b = 2 * ro.dot(ray.direction)
        c = ro.dot(ro) - self.radius**2
        radicand = b**2 - 4 * a * c
        if radicand < 0:
            hit_info.hit = False
            return hit_info
        hit_distance = min(quadratic_formula(a, b, c))
        if hit_distance > 0:
            hit_info.hit = True
            hit_info.hit_distance = hit_distance
            hit_info.hit_pos = ray.at(hit_distance)
            hit_info.hit_obj = self
            hit_info.hit_normal = self.get_normal(hit_info.hit_pos)
        else:
            hit_info.hit = False
        return hit_info

    def get_normal(self, pos):
        return (pos - self.pos).normalize()


class SimplePlane(Object):
    def __init__(self, material, height, name=None):
        super().__init__(material, name)
        self.height = height

    def intersect(self, ray):
        hit_info = HitInfo()
        y_diff = ray.origin.y - self.height
        if not (y_diff > 0 and ray.direction.y < 0) and not (
            y_diff < 0 and ray.direction.y > 0
        ):
            hit_info.hit = False
            return hit_info

        hit_info.hit = True
        hit_info.hit_distance = abs(
            y_diff / ray.direction.y * ray.direction.xz.magnitude()
        )
        hit_info.hit_normal = Vec3(0, 1, 0)
        hit_info.hit_obj = self
        hit_info.hit_pos = ray.origin + ray.direction * hit_info.hit_distance
        return hit_info

    def get_normal(self, pos):
        return Vec3(0, 1, 0)


class Ray:
    def __init__(
        self,
        origin,
        direction,
    ):
        self.origin = origin
        self.direction = direction

    def at(self, d):
        return self.origin + self.direction * d


class HitInfo:
    def __init__(
        self, hit=None, hit_normal=None, hit_pos=None, hit_obj=None, hit_distance=None
    ):
        self.hit = hit
        self.hit_normal = hit_normal
        self.hit_pos = hit_pos
        self.hit_obj = hit_obj
        self.hit_distance = hit_distance


class PostProcessing:
    ###### POST PROCESSING ######
    # Post processing pipeline:
    # 1. Fog
    # 2. Bloom
    # 3. Exposure
    # 4. White Balancing
    # 5. Contrast
    # 6. Brightness
    # 7. Color Filtering
    # 8. Saturation
    # 9. Tone Mapping
    # 9. Gamma Correction
    def __init__(self, exposure=1, brightness=0, contrast=1, saturation=1, gamma=1):
        self.exposure = exposure
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.gamma = gamma

    def process(self, img_arr):
        processed_img_arr = self.exposure_correction(img_arr)
        processed_img_arr = self.contrast_and_brightness_correction(processed_img_arr)
        processed_img_arr = self.saturation_correction(processed_img_arr)
        processed_img_arr = self.reinhardt_tonemapping(processed_img_arr)
        processed_img_arr = self.gamma_correction(processed_img_arr)
        return processed_img_arr

    def greyscale(self, img):
        return np.dot(img, np.array([0.299, 0.587, 0.114]))

    def exposure_correction(self, img):
        return img * self.exposure

    # def white_balancing(img, temp, tint):
    #     t1 = temp * 10 /6
    #     t2 = tint * 10 /6

    #     x

    def contrast_and_brightness_correction(self, img):
        return np.clip((self.contrast * (img - 0.5) + self.brightness + 0.5), 0, 1)

    def saturation_correction(self, img):
        grey = self.greyscale(img)[:, :, None] * np.array([[[1, 1, 1]]])
        return np.clip((lerp(grey, img, self.saturation)), 0, 1)

    def reinhardt_tonemapping(self, img, a=0.18, saturation=1.0):
        img = img + 1e-8
        Lw = self.greyscale(img)
        Lwa = np.exp(np.mean(np.log(Lw)))  # calculate the global adaptation luminance
        Lm = a / Lwa * Lw  # calculate the adapted luminance
        Ld = (
            Lm * (1 + Lm / (saturation**2)) / (1 + Lm)
        )  # apply the tonemapping function
        Ld_norm = Ld / np.max(Ld)  # normalize the output luminance to the range [0, 1]
        Ld_norm_3d = np.stack(
            (Ld_norm, Ld_norm, Ld_norm), axis=-1
        )  # create a 3-channel image from the luminance values
        output = np.clip(
            img / Lw[..., None] * Ld_norm_3d, 0, 1
        )  # apply the tonemapping to each pixel and clip the result to the range [0, 1]
        return output

    def gamma_correction(self, img):
        return np.power(img, self.gamma)


class Scene:
    def __init__(self) -> None:
        self.objects = []
        self.camera = Camera(Vec3(0), np.pi / 3, Vec3(0, 0, 1))
        self.environment_image = None

    def add_object(self, obj):
        self.objects.append(obj)

    def intersect(self, ray):
        hit_info = HitInfo(hit=False, hit_distance=float("inf"))
        for obj in self.objects:
            hit_info2 = obj.intersect(ray)
            if hit_info2.hit:
                if hit_info2.hit_distance < hit_info.hit_distance:
                    hit_info = hit_info2
        return hit_info

    def set_environment(self, img):
        self.environment_image = img

    def get_environment(self, ray):
        if self.environment_image == None:
            return Vec3(0)
        u, v = 0.5 + math.atan2(ray.direction.z, ray.direction.x) / (2 * np.pi), 1 - (
            0.5 + math.asin(ray.direction.y) / np.pi
        )
        x, y = int(self.environment_image.get_width() * u), int(
            self.environment_image.get_height() * v
        )
        c = Vec3(self.environment_image.get_at((x, y))[0:3]) / 255
        return c


class Renderer:
    def __init__(self, width, height, passes=1):
        self.WIDTH, self.HEIGHT = width, height
        self.RES = (width, height)
        self.ASPECT = height / width
        self._img_arr = np.zeros(shape=(width, height, 3))
        self.scene = Scene()
        self.max_bounces = 5

        self.break_ = [False]
        self.passes = [0]

        self.post_processing = PostProcessing()

    @property
    def img_arr(self):
        return self._img_arr / (self.passes[0] + 1)

    def reset_image(self):
        self._img_arr = np.zeros(shape=(self.WIDTH, self.HEIGHT, 3))

    def pixel_color(self, ray):
        ray_color = Vec3(1)
        incoming_light = Vec3(0)
        for i in range(self.max_bounces):
            hit_info = self.scene.intersect(ray)
            if hit_info.hit:
                ray.origin = hit_info.hit_pos

                specular_dir = ray.direction.reflect(hit_info.hit_normal)
                diffuse_dir = lerp(
                    random_hemisphere_sample(hit_info.hit_normal),
                    specular_dir,
                    hit_info.hit_obj.material.smoothness,
                ).normalize()
                is_specular = (
                    1
                    if hit_info.hit_obj.material.specular_probability > random.random()
                    else 0
                )

                ray.direction = specular_dir if is_specular else diffuse_dir

                emitted_light = (
                    hit_info.hit_obj.material.emission_color
                    * hit_info.hit_obj.material.emission_strength
                )
                incoming_light += ray_color.prod(emitted_light)
                ray_color = ray_color.prod(
                    (
                        hit_info.hit_obj.material.specular_color
                        if is_specular
                        else hit_info.hit_obj.material.color
                    ),
                )
            else:
                incoming_light += ray_color.prod(self.scene.get_environment(ray))
                break
        return incoming_light

    def _render(self, area=[0, 1, 0, 1]):
        for x in range(int(self.WIDTH * area[0]), int(self.WIDTH * area[1])):
            for y in range(int(self.HEIGHT * area[2]), int(self.HEIGHT * area[3])):
                u, v = (x + 0.5) / self.WIDTH * 2 - 1, (
                    y + 0.5
                ) / self.HEIGHT * 2 * self.ASPECT - self.ASPECT
                ray = self.scene.camera.get_ray(
                    u,
                    v,
                )
                self._img_arr[x, y] += self.pixel_color(ray)
        return self._img_arr

    def render(self, area=[0, 1, 0, 1]):
        t = Thread(target=self._render, args=(area,), daemon=True)
        t.start()

    def _render_threaded(self, grid_width, grid_height):
        w = 1 / grid_width
        h = 1 / grid_height
        self.passes = [0]
        self.break_ = [False]
        while True:
            if not self.break_[0]:
                np.random.seed(np.random.randint(0, 10000))
                threads = []
                for i in range(grid_width):
                    for j in range(grid_height):
                        threads.append(
                            Thread(
                                target=self._render,
                                args=([i * w, (i + 1) * w, j * h, (j + 1) * h],),
                                daemon=True,
                            )
                        )

                for t in threads:
                    t.start()

                for t in threads:
                    t.join()
                self.passes[0] += 1

    def render_threaded(self, grid_width, grid_height):
        t = Thread(
            target=(self._render_threaded), args=(grid_width, grid_height), daemon=True
        )
        t.start()

    def get_img_arr_raw(self):
        return self.img_arr

    def get_img_arr_post_processed(self):
        return self.post_processing.process(self.img_arr)

    def get_img_raw(self):
        return pg.surfarray.make_surface(np.clip(self.img_arr, 0, 1) * 255)

    def get_img_post_processed(self):
        return pg.surfarray.make_surface(self.get_img_arr_post_processed() * 255)


def get_scene1():
    scene = Scene()

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
    floor_mat.color = Vec3(0.8, 0.3, 0.3)
    floor_mat.emission_color = Vec3(1)
    floor_mat.specular_color = Vec3(1)
    floor_mat.emission_strength = 0
    floor_mat.smoothness = 0
    floor_mat.specular_probability = 0

    horse_mat = Material.default_material()
    horse_mat.color = Vec3(1)
    horse_mat.emission_color = Vec3(1)
    horse_mat.specular_color = Vec3(1)
    horse_mat.emission_strength = 0
    horse_mat.smoothness = 0
    horse_mat.specular_probability = 0

    horse = Mesh.load_from_obj_file("./assets/models/obj/chess_horse.obj")[0]
    horse.material = horse_mat
    horse.set_origin(Vec3(0, 0, 50))
    horse.set_scale(Vec3(3))

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
        Vec3(0, 3, 0), np.pi / 10, Vec3(0, 100, 1000), dof_strength=0.001, dof_dist=30
    )

    for obj in objects:
        scene.add_object(obj)

    scene.camera = camera
    scene.set_environment(pg.image.load("./assets/skyboxes/skybox1.png"))

    return scene


def main():
    pg.init()
    WIDTH, HEIGHT = 500, 500
    RES = (WIDTH, HEIGHT)
    screen = pg.display.set_mode(RES)
    clock = pg.time.Clock()

    renderer = Renderer(int(WIDTH / 5), int(HEIGHT / 5))

    renderer.scene = get_scene1()
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
