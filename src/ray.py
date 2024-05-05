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
