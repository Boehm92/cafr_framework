import madcad as mdc


class CsgOperation:
    def __init__(self, target_primitive, sub_primitive):
        self.target_primitive = target_primitive
        self.sub_primitive = sub_primitive

    def difference(self):
        _new_3d_model = mdc.difference(self.target_primitive, self.sub_primitive)
        _new_3d_model.mergeclose()
        _new_3d_model = mdc.segmentation(_new_3d_model)

        return _new_3d_model
