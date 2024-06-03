from typing import List

from src.hit_info import HitInfo
from src.material import Material
from src.objects.base import ObjectBase
from src.ray import Ray
from src.util import Vec3


class Mesh(ObjectBase):
    def __init__(
        self,
        material: Material,
        origin: Vec3,
        vertices: List[Vec3],
        tris: List[int],
        scale: Vec3 = None,
        rotation: Vec3 = None,
    ):
        super().__init__(material, origin)
        self.vertices: List[Vec3] = vertices
        self.tris: List[int] = tris
        self.transformed_vertices = vertices

        self.scale: Vec3 = scale if scale else Vec3(1)
        self.rotation: Vec3 = rotation if rotation else Vec3(0)

        self.update_transformed_vertices()

    def get_scaled_vertices(self, points) -> List[Vec3]:
        return [p.prod(self.scale) for p in points]

    def get_translated_vertices(self, points) -> List[Vec3]:
        return [p + self.origin for p in points]

    def get_rotated_vertices(self) -> List[Vec3]:
        rotated = []
        for v in self.vertices:
            v.rotate(
                self.rotation.x,
            )
        return self.vertices

    def get_transformed_vertices(self) -> List[Vec3]:
        vertices = self.get_scaled_vertices(self.vertices)
        # vertices = self.get_rotated_vertices(vertices)
        vertices = self.get_translated_vertices(vertices)
        return vertices

    def update_transformed_vertices(self) -> None:
        self.transformed_vertices = self.get_transformed_vertices()

    def set_scale(self, scale: Vec3) -> None:
        self.scale = scale
        self.update_transformed_vertices()

    def set_origin(self, origin: Vec3) -> None:
        self.origin = origin
        self.update_transformed_vertices()

    def set_rotation(self, rotation: Vec3) -> None:
        self.update_transformed_vertices()

    def intersect(self, ray: Ray) -> List[HitInfo]:
        hits = []
        for face in self.tris:
            info = self.intersect_tri(face, ray)
            if info.hit:
                hits.append(info)
        return hits

    def intersect_tri(self, tri: List[int], ray: Ray) -> HitInfo:
        point_a = self.transformed_vertices[tri[0]]
        point_b = self.transformed_vertices[tri[1]]
        point_c = self.transformed_vertices[tri[2]]

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

        info = HitInfo(obj=self)
        if determinant >= 1e-6 and dst >= 0 and u >= 0 and v >= 0 and w >= 0:
            info.hit = True
            info.depth = dst
            info.pos = ray.at(dst)
            info.normal = normal.normalize()
        else:
            info.hit = False
        return info

    @classmethod
    def load_from_obj_file(self, path) -> "Mesh":
        with open(path, "r") as f:
            vertices = []
            tris = []
            vertex_index_offset = 0
            meshes = []
            for line in f.readlines():
                if line.startswith("o "):
                    if meshes:
                        meshes[-1].vertices = vertices
                        meshes[-1].tris = tris
                        meshes[-1].update_transformed_vertices()
                        vertex_index_offset += len(vertices)
                        vertices = []
                        tris = []
                    m = Mesh(Material.default_material(), [], [])
                    m.name = line.split(" ")[-1][:-1]
                    meshes.append(m)
                elif line.startswith("v "):
                    coords = line.split(" ")[1:]
                    coords = [float(v) for v in coords]
                    vertices.append(Vec3(*coords))
                elif line.startswith("f "):
                    indecies = line.split(" ")[1:]
                    indecies = [
                        int(index.split("/")[0]) - 1 - vertex_index_offset
                        for index in indecies
                    ]
                    tris.append(indecies)
            else:
                meshes[-1].vertices = vertices
                meshes[-1].tris = tris
                meshes[-1].update_transformed_vertices()
        return meshes
