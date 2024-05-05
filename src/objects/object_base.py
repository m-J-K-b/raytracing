class ObjectBase:
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
