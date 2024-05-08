from src.hit_info import HitInfo
from src.material import Material
from src.objects.object_base import ObjectBase
from src.util import Vec3


class Mesh(ObjectBase):
    def __init__(self, material, points, faces):
        super().__init__(material)
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
