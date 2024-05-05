import math
import random
import sys
from copy import copy
from threading import Thread
import multiprocessing as mp

import numpy as np
import pygame as pg


class Vec3(pg.Vector3):
    def __init_subclass__(cls):
        return super().__init_subclass__()

    def prod(self, other):
        return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)


class Camera:
    def __init__(self, pos, fov, lookat, dof_strength = 0, dof_dist = 1):
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
        ray_pos = self.pos + self.right * ((random.random() * 2 - 1) * self.dof_strength) +  self.up * ((random.random() * 2 - 1) * self.dof_strength)
        ray_dir = (dof_target - ray_pos).normalize()
        return Ray(
            ray_pos, ray_dir
        )


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
        if not (y_diff > 0 and ray.direction.y < 0) and not (y_diff < 0 and ray.direction.y > 0):
            hit_info.hit = False
            return hit_info

        hit_info.hit = True            
        hit_info.hit_distance = abs(y_diff / ray.direction.y * ray.direction.xz.magnitude())
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


def smoothstep(v, minv, maxv):
    if v < minv:
        return 0
    elif v > maxv:
        return 1

    v = (v - minv) / (maxv - minv)

    return v * v * (3 - 2 * v)


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


def intersect(ray, objects):
    hit_info = HitInfo(hit=False, hit_distance=float("inf"))
    for obj in objects:
        hit_info2 = obj.intersect(ray)
        if hit_info2.hit:
            if hit_info2.hit_distance < hit_info.hit_distance:
                hit_info = hit_info2
    return hit_info

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


def greyscale(img):
    return np.dot(img, np.array([0.299, 0.587, 0.114]))

def exposure_correction(img, exposure):
    return img * exposure

# def white_balancing(img, temp, tint):
#     t1 = temp * 10 /6
#     t2 = tint * 10 /6

#     x 

def contrast_and_brightness_correction(img, contrast, brightness):
    return np.clip((contrast * (img - 0.5) + brightness + 0.5), 0, 1)

def saturation_correction(img, saturation):
    grey = greyscale(img)[:, :, None] * np.array([[[1, 1, 1]]])
    return np.clip((lerp(grey, img, saturation)), 0, 1)

def reinhardt_tonemapping(img, a=0.18, saturation=1.0):
    Lw = greyscale(img)
    Lwa = np.exp(np.mean(np.log(Lw)))  # calculate the global adaptation luminance
    Lm = a / Lwa * Lw  # calculate the adapted luminance
    Ld = Lm * (1 + Lm / (saturation**2)) / (1 + Lm)  # apply the tonemapping function
    Ld_norm = Ld / np.max(Ld)  # normalize the output luminance to the range [0, 1]
    Ld_norm_3d = np.stack(
        (Ld_norm, Ld_norm, Ld_norm), axis=-1
    )  # create a 3-channel image from the luminance values
    output = np.clip(
        img / Lw[..., None] * Ld_norm_3d, 0, 1
    )  # apply the tonemapping to each pixel and clip the result to the range [0, 1]
    return output
    
def gamma_correction(img, power=0.5):
    return np.power(img, power)


# sun_light_dir = Vec3(1, -1, 1).normalize() * -1
# sun_focus = 50
# sun_intensity = 5
# horizon, zenith, ground = (
#     Vec3(100, 120, 225) / 255,
#     Vec3(150, 150, 240) / 255,
#     Vec3(0),
# )
# # horizon, zenith, ground = Vec3(0), Vec3(0), Vec3(0)
# def get_environment(ray):
#     t = smoothstep(ray.direction.y, 0, 0.4) * 0.35
#     c = lerp(zenith, horizon, t)
#     sun = max(0, ray.direction.dot(sun_light_dir)) ** sun_focus * sun_intensity

#     gt = smoothstep(ray.direction.y, -0.01, 0)
#     # sm = gt >= 1
#     return lerp(ground, c, gt) + Vec3(sun * gt)


# def get_environment(ray):
#     return Vec3(0)


environment_image = None


# def get_environment(ray):
#     if environment_image == None:
#         return Vec3(0)
#     u, v = 0.5 + math.atan2(ray.direction.z, ray.direction.x) / (2 * np.pi), 1 - (
#         0.5 + math.asin(ray.direction.y) / np.pi
#     )
#     x, y = int(environment_image.get_width() * u), int(
#         environment_image.get_height() * v
#     )
#     c = Vec3(environment_image.get_at((x, y))[0:3]) / 255
#     return c
def get_environment(ray):
    return Vec3(0)
  


def pixel_color(ray, objects, max_bounces):
    ray_color = Vec3(1)
    incoming_light = Vec3(0)
    for i in range(max_bounces):
        hit_info = intersect(ray, objects)
        if hit_info.hit:
            # return hit_info.hit_obj.material.color
            ray.origin = hit_info.hit_pos

            # diffuse_dir = (random_hemisphere_sample(hit_info.hit_normal) + hit_info.hit_normal).normalize()
            specular_dir = ray.direction.reflect(hit_info.hit_normal)
            diffuse_dir = lerp(random_hemisphere_sample(hit_info.hit_normal), specular_dir, hit_info.hit_obj.material.smoothness).normalize()
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
                hit_info.hit_obj.material.specular_color if is_specular else hit_info.hit_obj.material.color,
            )
        else:
            incoming_light += ray_color.prod(get_environment(ray))
            break
    return incoming_light


def render(arr, objects, camera, max_bounces=3, area=[0, 1, 0, 1], eps=0.01):
    half_eps = eps / 2
    WIDTH, HEIGHT = arr.shape[0], arr.shape[1]
    aspect = HEIGHT / WIDTH
    for x in range(int(WIDTH * area[0]), int(WIDTH * area[1])):
        for y in range(int(HEIGHT * area[2]), int(HEIGHT * area[3])):
            u, v = (x + 0.5) / WIDTH * 2 - 1, (y + 0.5) / HEIGHT * 2 * aspect - aspect
            ray = camera.get_ray(
                u + random.random() * half_eps - eps,
                v + random.random() * half_eps - eps,
            )
            arr[x, HEIGHT - y - 1] += pixel_color(
                ray, objects, max_bounces=max_bounces + 1
            )
    return arr


def render_threaded(
    grid_width, grid_height, arr, objects, camera, max_bounces, break_, passes
):
    while True:
        if not break_[0]:
            np.random.seed(np.random.randint(0, 10000))
            threads = []
            w = 1 / grid_width
            h = 1 / grid_height
            for i in range(grid_width):
                for j in range(grid_height):
                    threads.append(
                        Thread(
                            target=render,
                            args=(
                                arr,
                                objects,
                                camera,
                                max_bounces,
                                [i * w, (i + 1) * w, j * h, (j + 1) * h],
                            ),
                            daemon=True,
                        )
                    )

            for t in threads:
                t.start()

            for t in threads:
                t.join()
            passes[0] += 1


def main():
    global environment_image
    pg.init()
    random.seed(6)
    WIDTH, HEIGHT = 1080 / 2, 1920 / 2
    RES = (WIDTH, HEIGHT)
    PX_RES = 0.5
    SURF_WIDTH, SURF_HEIGHT = int(WIDTH / PX_RES), int(HEIGHT / PX_RES)
    print(SURF_WIDTH, SURF_HEIGHT)
    SURF_RES = (SURF_WIDTH, SURF_HEIGHT)

    screen = pg.display.set_mode(RES)
    surface = pg.Surface(SURF_RES)
    img_arr = np.zeros(shape=(SURF_WIDTH, SURF_HEIGHT, 3), dtype=np.float64)
    clock = pg.time.Clock()

    # environment_image = pg.image.load("./assets/skyboxes/skybox1.png").convert_alpha()
    # environment_image = None
    # environment_image.fill((1, 1, ))
    
    # camera = Camera(Vec3(-4.3, 0, 0), np.pi / 2, Vec3(0, 0, 0))
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
    
    # sphere_mat1 = Material.default_material()
    # sphere_mat1.color = Vec3(0.5, 1, 0.5)
    # sphere_mat1.emission_strength = 0
    # sphere_mat1.smoothness = 0.1
    # sphere_mat1.emission_color = Vec3(1)
    
    # sphere_mat2 = Material.default_material()
    # sphere_mat2.color = Vec3(1, 1, 0.5)
    # sphere_mat2.emission_strength = 0
    # sphere_mat2.smoothness = 0.1
    
    big_sphere_mat = Material.default_material()
    big_sphere_mat.color = Vec3(252, 172, 68) / 255
    big_sphere_mat.emission_strength = 1
    big_sphere_mat.emission_color = Vec3(252, 172, 60) / 255
    big_sphere_mat.smoothness = 0
    big_sphere_mat.specular_probability = 0
    big_sphere_mat.specular_color = Vec3(1)
    # big_sphere_mat.emission_color = Vec3(1)

    
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
        horse

    ] + [Sphere(random.choice(materials), Vec3(6 * (1 if i % 2 == 0 else -1), 3, (i // 2) * 20 + 10), 3) for i in range(0, 100)]
    # camera = Camera(Vec3(0, 20, -5), np.pi / 3, sum_, dof_dist=5, dof_strength=0)
    camera = Camera(Vec3(0, 3, 0), np.pi / 10, Vec3(0, 100, 1000), dof_strength=0.5, dof_dist=50)
    
    # random.seed(10)
    # for i in range(30000):
    #     environment_image.set_at((random.randint(0, 3000), random.randint(0, 3000)), (255, 255, 255))
    
    break_ = [False]
    passes = [0]

    # t = Thread(
    #     target=render_threaded,
    #     args=(8, 16, surface_arr, objects, camera, 5, break_, passes),
    #     daemon=True,
    # )
    t = Thread(
        target=render_threaded,
        args=(1, 1, img_arr, objects, camera, 5, break_, passes),
        daemon=True,
    )
    t.start()

    sample_num = 50

    while True:
        post_processed_img_arr = exposure_correction(img_arr, 1.3)
        post_processed_img_arr = contrast_and_brightness_correction(post_processed_img_arr, 1, .05)
        post_processed_img_arr = saturation_correction(post_processed_img_arr, 1.2)
        post_processed_img_arr = reinhardt_tonemapping(post_processed_img_arr + 1e-4)
        post_processed_img_arr = gamma_correction(post_processed_img_arr, power=1.1)
        
        img = pg.surfarray.make_surface(img_arr * 255)
        img_post_processed = pg.surfarray.make_surface(post_processed_img_arr * 255)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    break_[0] = not break_[0]

                elif event.key == pg.K_s:
                    pg.image.save(surface, f"./image@{hash(random.random())}.png")

        scaled_surface = pg.transform.scale(surface, RES)
        screen.blit(scaled_surface, (0, 0))

        if passes[0] == sample_num - 1:
            break_[0] = True
        
        if passes[0] == sample_num:
            # surface_arr2 = np.minimum(1, surface_arr / passes) * 255
            # surface = pg.surfarray.make_surface(surface_arr2)
            # surface = pg.transform.scale(surface, RES)
            pg.image.save(pg.surfarray.surface, f"./image@{hash(random.random())}.png")
            pg.image.save(post_processed_img_arr, f"./image_post_processed@{hash(random.random())}.png")
            pg.quit()
            sys.exit()

        pg.display.set_caption(f"Iteration: {passes}, {break_[0]}")

        pg.display.update()
        clock.tick(60)


if __name__ == "__main__":
    main()
