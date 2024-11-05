from dataclasses import dataclass, field

from src.objects import ObjectBase
from src.util import Vec3


@dataclass
class HitInfo:
    hit: bool = False
    depth: float = 0
    pos: Vec3 = field(default_factory=Vec3)
    normal: Vec3 = field(default_factory=Vec3)
    obj: ObjectBase = None  # type: ignore
