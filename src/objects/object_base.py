class ObjectBase:
    def __init__(self, material):
        self.material = material

    def update(self, t):
        for animation in self.animations:
            animation.update(t)

    def intersect(self, ray):
        raise NotImplementedError()
