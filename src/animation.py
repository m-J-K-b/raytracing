from src.util import exponential_interpolation, lerp, smooth_interpolation


class Animation:
    def __init__(self, obj_reference, attribute_name, animation_keys=None):
        self.obj_reference = obj_reference
        self.attribute_name = attribute_name
        self.animation_keys = animation_keys if animation_keys != None else []
        self.animation_key_num = len(animation_keys)
        self.initial_attribute_value = getattr(obj_reference, attribute_name)

        self.interpolation_functions = [
            lerp,
            exponential_interpolation,
            smooth_interpolation,
        ]

    def get_interpolated_value(self, v1, v2, t, interpolation_type, interpolation_args):
        return self.interpolation_functions[interpolation_type](
            v1, v2, t, *interpolation_args
        )

    def get_animated_value(self, t):
        current_value = self.initial_attribute_value

        for i, (time_stamp, value, interpolation_type) in enumerate(
            self.animation_keys
        ):
            if time_stamp >= t:
                if i < self.animation_key_num - 1:
                    next_time_stamp, next_value, _ = self.animation_keys[i + 1]
                    current_value = self.get_interpolated_value(
                        value,
                        next_value,
                        (t - time_stamp) / (next_time_stamp - time_stamp),
                        interpolation_type,
                    )
                else:
                    current_value = value
                break

        return current_value

    def update(self, t):
        setattr(self.obj_reference, self.attribute_name, self.get_animated_value(t))


class AnimationManager:
    def __init__(self):
        self.animations = []

    def add_animation(self, animation: Animation):
        self.animations.append(animation)

    def update(self, t):
        for animation in self.animations:
            animation.update(t)
