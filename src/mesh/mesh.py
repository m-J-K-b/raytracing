from __future__ import annotations

from typing import List

from src.core import HitInfo, Material, Ray, Vec3
from src.mesh.base import ObjectBase


class Mesh(ObjectBase):
    """
    A mesh object that supports basic transformations, ray intersection testing,
    and loading from an OBJ file.
    """

    def __init__(
        self,
        material: Material,
        origin: Vec3,
        vertices: List[Vec3],
        triangles: List[List[int]],
        scale: Vec3 | None = None,
        rotation: Vec3 | None = None,
    ):
        super().__init__(material, origin)
        self.vertices: List[Vec3] = vertices
        self.triangles: List[List[int]] = triangles

        # Transformation parameters
        self.scale: Vec3 = scale if scale else Vec3(1)
        self.rotation: Vec3 = rotation if rotation else Vec3(0)

        # Cached transformed vertices and bounding box
        self.transformed_vertices: List[Vec3] = []
        self.bounding_box: tuple[Vec3, Vec3] | None = None

        self.update_transformed_vertices()

    def update_transformed_vertices(self) -> None:
        """
        Update transformed vertices based on scale, rotation, and translation.
        Also updates the bounding box.
        """
        self.transformed_vertices = self.get_transformed_vertices()
        self.update_bounding_box()

    def update_bounding_box(self) -> None:
        """
        Compute the axis-aligned bounding box (AABB) from the transformed vertices.
        """
        if not self.transformed_vertices:
            self.bounding_box = None
            return
        min_x = min(v.x for v in self.transformed_vertices)
        min_y = min(v.y for v in self.transformed_vertices)
        min_z = min(v.z for v in self.transformed_vertices)
        max_x = max(v.x for v in self.transformed_vertices)
        max_y = max(v.y for v in self.transformed_vertices)
        max_z = max(v.z for v in self.transformed_vertices)
        self.bounding_box = (Vec3(min_x, min_y, min_z), Vec3(max_x, max_y, max_z))

    def intersect_aabb(self, ray: Ray) -> bool:
        """
        Perform a ray-AABB intersection test using the slab method.
        Returns True if the ray intersects the bounding box.
        """
        if self.bounding_box is None:
            return False

        # Inverse ray directions for the slab test.
        inv_dir = Vec3(
            1 / ray.direction.x if ray.direction.x != 0 else 1e10,
            1 / ray.direction.y if ray.direction.y != 0 else 1e10,
            1 / ray.direction.z if ray.direction.z != 0 else 1e10,
        )

        tmin_vec = (self.bounding_box[0] - ray.origin).elementwise() * inv_dir
        tmax_vec = (self.bounding_box[1] - ray.origin).elementwise() * inv_dir

        t1 = Vec3(
            min(tmin_vec.x, tmax_vec.x),
            min(tmin_vec.y, tmax_vec.y),
            min(tmin_vec.z, tmax_vec.z),
        )
        t2 = Vec3(
            max(tmin_vec.x, tmax_vec.x),
            max(tmin_vec.y, tmax_vec.y),
            max(tmin_vec.z, tmax_vec.z),
        )

        t_near = max(t1.x, t1.y, t1.z)
        t_far = min(t2.x, t2.y, t2.z)
        return t_far >= max(t_near, 0)

    def intersect(self, ray: Ray) -> List[HitInfo]:
        """
        Check for intersections between the ray and all triangles in the mesh.
        Uses an AABB test for early rejection.
        """
        hits: List[HitInfo] = []
        if not self.intersect_aabb(ray):
            return hits

        for face in self.triangles:
            info = self.intersect_tri(face, ray)
            if info.hit:
                hits.append(info)
        return hits

    def intersect_tri(self, tri: List[int], ray: Ray) -> HitInfo:
        """
        Intersect a single triangle (specified by three indices) with the ray.
        """
        point_a = self.transformed_vertices[tri[0]]
        point_b = self.transformed_vertices[tri[1]]
        point_c = self.transformed_vertices[tri[2]]

        edge_ab = point_b - point_a
        edge_ac = point_c - point_a
        normal = edge_ab.cross(edge_ac)

        # Use a small epsilon to avoid division by zero issues.
        determinant = -ray.direction.dot(normal)
        if abs(determinant) < 1e-6:
            return HitInfo(obj=self, hit=False)
        inv_det = 1 / determinant

        ao = ray.origin - point_a
        dao = ao.cross(ray.direction)

        dst = normal.dot(ao) * inv_det
        u = edge_ac.dot(dao) * inv_det
        v = -edge_ab.dot(dao) * inv_det
        w = 1 - u - v

        info = HitInfo(obj=self)
        if dst >= 0 and u >= 0 and v >= 0 and w >= 0:
            info.hit = True
            info.depth = dst
            info.pos = ray.at(dst)
            info.normal = normal.normalize()
        else:
            info.hit = False
        return info

    def get_scaled_vertices(self, points: List[Vec3]) -> List[Vec3]:
        """
        Scale the given vertices by the mesh's scale factor.
        """
        return [p.prod(self.scale) for p in points]

    def get_rotated_vertices(self, vertices: List[Vec3]) -> List[Vec3]:
        """
        Rotate the vertices around the x, y, and z axes by the mesh's rotation angles.
        """
        rotated = []
        for v in vertices:
            vr = v.rotate(self.rotation.x, Vec3(1, 0, 0))
            vr = vr.rotate(self.rotation.y, Vec3(0, 1, 0))
            vr = vr.rotate(self.rotation.z, Vec3(0, 0, 1))
            rotated.append(vr)
        return rotated

    def get_translated_vertices(self, points: List[Vec3]) -> List[Vec3]:
        """
        Translate the vertices by the mesh's origin.
        """
        return [p + self.origin for p in points]

    def get_transformed_vertices(self) -> List[Vec3]:
        """
        Apply scaling, rotation, and translation to the original vertices.
        """
        vertices = self.get_scaled_vertices(self.vertices)
        vertices = self.get_rotated_vertices(vertices)
        vertices = self.get_translated_vertices(vertices)
        return vertices

    def set_scale(self, scale: Vec3) -> None:
        """
        Set a new scale and update the transformed vertices.
        """
        self.scale = scale
        self.update_transformed_vertices()

    def set_origin(self, origin: Vec3) -> None:
        """
        Set a new origin (translation) and update the transformed vertices.
        """
        self.origin = origin
        self.update_transformed_vertices()

    def set_origin_to_center_of_mass(self) -> None:
        """
        Adjust the mesh so that its origin is at the center of mass.
        """
        com = Vec3(0)
        for vertex in self.vertices:
            com += vertex
        com /= len(self.vertices)
        self.vertices = [vertex - com for vertex in self.vertices]
        self.origin += com
        self.update_transformed_vertices()

    def set_rotation(self, rotation: Vec3) -> None:
        """
        Set a new rotation and update the transformed vertices.
        """
        self.rotation = rotation
        self.update_transformed_vertices()

    @classmethod
    def load_from_obj_file(cls, path: str) -> List[ObjectBase]:
        """
        Load one or more meshes from an OBJ file.
        Supports multiple object definitions and triangulates faces if necessary.
        """
        with open(path, "r") as f:
            vertices = []
            triangles = []
            vertex_index_offset = 0
            meshes: List[Mesh] = []
            for line in f:
                line = line.strip()
                if line.startswith("o "):
                    if meshes:
                        meshes[-1].vertices = vertices
                        meshes[-1].triangles = triangles
                        meshes[-1].update_transformed_vertices()
                        vertex_index_offset += len(vertices)
                        vertices = []
                        triangles = []
                    m = Mesh(Material.default_material(), Vec3(0), [], [])
                    m.name = line.split(" ", 1)[1]
                    meshes.append(m)
                elif line.startswith("v "):
                    coords = list(map(float, line.split()[1:]))
                    vertices.append(Vec3(*coords))
                elif line.startswith("f "):
                    indices = [
                        int(part.split("/")[0]) - 1 - vertex_index_offset
                        for part in line.split()[1:]
                    ]
                    if len(indices) >= 3:
                        # Triangulate the face if it has more than three vertices.
                        for i in range(1, len(indices) - 1):
                            triangles.append([indices[0], indices[i], indices[i + 1]])
            if meshes:
                meshes[-1].vertices = vertices
                meshes[-1].triangles = triangles
                meshes[-1].update_transformed_vertices()
            return meshes
