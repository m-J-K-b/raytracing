from dataclasses import dataclass

from src.util import Vec3


@dataclass
class HitInfo:
    depth: float = None
    normal: Vec3 = None
    pos: Vec3 = None
    obj: int = None
