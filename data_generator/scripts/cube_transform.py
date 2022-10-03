import numpy as np
from madcad import *


def rotate_model_randomly(model: Mesh):

    rotation_axis = np.random.choice(['None', 'x', 'y', 'z', 'x', 'y', 'z'])

    rotation = np.random.choice([-90, 90])

    if rotation_axis == 'None':
        return model

    if rotation_axis == 'x' and rotation == 90:
        model = model.transform(quat(radians(rotation)*vec3(1, 0, 0)))
        model = model.transform(vec3(0, 10, 0))

    if rotation_axis == 'x' and rotation == -90:
        model = model.transform(quat(radians(rotation)*vec3(1, 0, 0)))
        model = model.transform(vec3(0, 0, 10))

    if rotation_axis == 'y' and rotation == 90:
        model = model.transform(quat(radians(rotation)*vec3(0, 1, 0)))
        model = model.transform(vec3(0, 0, 10))

    if rotation_axis == 'y' and rotation == -90:
        model = model.transform(quat(radians(rotation)*vec3(0, 1, 0)))
        model = model.transform(vec3(10, 0, 0))

    if rotation_axis == 'z' and rotation == 90:
        model = model.transform(quat(radians(rotation)*vec3(0, 0, 1)))
        model = model.transform(vec3(10, 0, 0))

    if rotation_axis == 'z' and rotation == -90:
        model = model.transform(quat(radians(rotation)*vec3(0, 0, 1)))
        model = model.transform(vec3(0, 10, 0))

    return model
