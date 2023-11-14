import numpy as np
import madcad as mdc


class Cube:
    def __init__(self, side_length, position):
        self.side_length = side_length
        self.position = position

    def transform(self):
        return mdc.brick(width=mdc.vec3(self.side_length)).transform(self.position)

    @staticmethod
    def rotate_model_randomly(model):

        rotation_axis = np.random.choice(['None', 'x', 'y', 'z'])

        rotation = np.random.choice([-90, 90])

        if rotation_axis == 'None':
            return model

        if rotation_axis == 'x' and rotation == 90:
            model = model.transform(mdc.quat(mdc.radians(rotation) * mdc.vec3(1, 0, 0)))
            model = model.transform(mdc.vec3(0, 10, 0))

        if rotation_axis == 'x' and rotation == -90:
            model = model.transform(mdc.quat(mdc.radians(rotation) * mdc.vec3(1, 0, 0)))
            model = model.transform(mdc.vec3(0, 0, 10))

        if rotation_axis == 'y' and rotation == 90:
            model = model.transform(mdc.quat(mdc.radians(rotation) * mdc.vec3(0, 1, 0)))
            model = model.transform(mdc.vec3(0, 0, 10))

        if rotation_axis == 'y' and rotation == -90:
            model = model.transform(mdc.quat(mdc.radians(rotation) * mdc.vec3(0, 1, 0)))
            model = model.transform(mdc.vec3(10, 0, 0))

        if rotation_axis == 'z' and rotation == 90:
            model = model.transform(mdc.quat(mdc.radians(rotation) * mdc.vec3(0, 0, 1)))
            model = model.transform(mdc.vec3(10, 0, 0))

        if rotation_axis == 'z' and rotation == -90:
            model = model.transform(mdc.quat(mdc.radians(rotation) * mdc.vec3(0, 0, 1)))
            model = model.transform(mdc.vec3(0, 10, 0))

        return model
