from dataclasses import dataclass

from src.util import Vec3
from src.objects import ObjectBase


@dataclass
class HitInfo:
    hit: bool = False
    depth: float = 0
    pos: Vec3 = Vec3(0)
    normal: Vec3 = Vec3(0)
    obj: ObjectBase = None
