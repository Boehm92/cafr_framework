import os
import numpy as np
import numpy.random
import madcad as mdc


class MachiningFeature:
    def __init__(self, model, machining_feature, min_scale, max_scale, min_depth, max_depth):
        self.model = model
        self.machining_feature = machining_feature
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_depth = max_depth
        self.min_depth = min_depth

        self.machining_feature_functions = [self.o_ring, self.trough_hole,
                                            self.blind_hole, self.triangular_passage,
                                            self.rectangular_passage, self.circular_trough_slot,
                                            self.triangular_trough_slot, self.rectangular_trough_slot,
                                            self.rectangular_blind_slot, self.triangular_pocket,
                                            self.rectangular_pocket, self.circular_end_pocket,
                                            self.triangular_blind_step, self.circular_blind_step,
                                            self.rectangular_blind_step, self.rectangular_trough_step,
                                            self.two_side_through_step, self.slanted_through_step,
                                            self.chamfer, self.round,
                                            self.vertical_circular_end_blind_slot,
                                            self.horizontal_circular_end_blind_slot,
                                            self.six_side_passage, self.six_side_pocket]

    def apply_feature(self):
        return self.machining_feature_functions[self.machining_feature]()

    def o_ring(self):
        _outside_ring = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/O_Ring.stl')
        _inside_ring = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/O_Ring.stl')
        _outside_ring.mergeclose()
        _inside_ring.mergeclose()
        _outside_ring = mdc.segmentation(_outside_ring)
        _inside_ring = mdc.segmentation(_inside_ring)

        _outside_diameter = np.random.uniform(self.min_scale, self.max_scale)
        _inside_diameter = np.random.uniform(self.min_scale, _outside_diameter - 1)
        _depth = np.random.uniform(self.max_depth, self.max_depth)
        _diameter_difference = (_outside_diameter / 2) - (_inside_diameter / 2)

        _outside_ring = _outside_ring.transform(
            mdc.mat3((_outside_diameter / 9), (_outside_diameter / 9), (_depth / 9)))
        _inside_ring = _inside_ring.transform(mdc.mat3((_inside_diameter / 9), (_inside_diameter / 9), (_depth / 9)))
        _inside_ring = _inside_ring.transform(mdc.vec3(_diameter_difference, _diameter_difference, 0))

        _o_ring = mdc.difference(_outside_ring, _inside_ring)

        _position_x = np.random.uniform(0.5, 9.5 - _outside_diameter)
        _position_y = np.random.uniform(0.5, 9.5 - _outside_diameter)
        _o_ring = _o_ring.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _o_ring)

        return updated_model

    def trough_hole(self):
        _trough_hole = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Trough_Hole.stl')

        _trough_hole.mergeclose()
        _trough_hole = mdc.segmentation(_trough_hole)

        _diameter = np.random.uniform(self.min_scale, self.max_scale)
        _position_x = np.random.uniform(0.5, 9.5 - _diameter)
        _position_y = np.random.uniform(0.5, 9.5 - _diameter)

        _trough_hole = _trough_hole.transform(mdc.mat3((_diameter / 9), (_diameter / 9), 1))
        _trough_hole = _trough_hole.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _trough_hole)

        return updated_model

    def blind_hole(self):
        _blind_hole = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Blind_Hole.stl')

        _blind_hole.mergeclose()
        _blind_hole = mdc.segmentation(_blind_hole)

        _diameter = np.random.uniform(self.min_scale, self.max_scale)
        _depth = np.random.uniform(self.max_depth, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _diameter)
        _position_y = np.random.uniform(0.5, 9.5 - _diameter)

        _blind_hole = _blind_hole.transform(mdc.mat3((_diameter / 9), (_diameter / 9), (_depth / 9)))
        _blind_hole = _blind_hole.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _blind_hole)

        return updated_model

    def triangular_passage(self):
        _triangular_passage = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Triangular_Passage.stl')

        _triangular_passage.mergeclose()
        _triangular_passage = mdc.segmentation(_triangular_passage)

        _length = np.random.uniform(self.min_scale, self.max_scale)
        _width = np.random.uniform(self.min_scale, self.max_scale)
        _position_x = np.random.uniform(0.5, 9.5 - _length)
        _position_y = np.random.uniform(0.5, 9.5 - _width)

        _triangular_passage = _triangular_passage.transform(mdc.mat3((_length / 9), (_width / 9), 1))
        _triangular_passage = _triangular_passage.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _triangular_passage)

        return updated_model

    def rectangular_passage(self):
        _rectangular_passage = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Rectangular_Passage.stl')

        _rectangular_passage.mergeclose()
        _rectangular_passage = mdc.segmentation(_rectangular_passage)

        _length = np.random.uniform(self.min_scale, self.max_scale)
        _width = np.random.uniform(self.min_scale, self.max_scale)
        _position_x = np.random.uniform(0.5, 9.5 - _length)
        _position_y = np.random.uniform(0.5, 9.5 - _width)

        _rectangular_passage = _rectangular_passage.transform(mdc.mat3((_length / 9), (_width / 9), 1))
        _rectangular_passage = _rectangular_passage.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _rectangular_passage)

        return updated_model

    def circular_trough_slot(self):
        _circular_trough_slot = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Circular_Trough_Slot.stl')

        _circular_trough_slot.mergeclose()
        _circular_trough_slot = mdc.segmentation(_circular_trough_slot)

        _radius = np.random.uniform(1, self.max_scale)
        _position_x = np.random.uniform(0.5, (9.5 - _radius))

        _circular_trough_slot = _circular_trough_slot.transform(mdc.mat3(_radius / 9, _radius / 9, 1))
        _circular_trough_slot = _circular_trough_slot.transform(mdc.vec3(_position_x, -0.1, -0.01))

        updated_model = mdc.difference(self.model, _circular_trough_slot)

        return updated_model

    def triangular_trough_slot(self):
        _triangular_trough_slot = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Triangular_Trough_Slot.stl')

        _triangular_trough_slot.mergeclose()
        _triangular_trough_slot = mdc.segmentation(_triangular_trough_slot)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)
        _position_x = np.random.uniform(0.5, (9.5 - _width))

        _triangular_trough_slot = _triangular_trough_slot.transform(mdc.mat3(_width / 9, _depth / 9, 1))
        _triangular_trough_slot = _triangular_trough_slot.transform(mdc.vec3(_position_x, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _triangular_trough_slot)

        return updated_model

    def rectangular_trough_slot(self):
        _rectangular_trough_slot = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Rectangular_Trough_Slot.stl')

        _rectangular_trough_slot.mergeclose()
        _rectangular_trough_slot = mdc.segmentation(_rectangular_trough_slot)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)
        _position_x = np.random.uniform(0.5, (9.5 - _width))

        _rectangular_trough_slot = _rectangular_trough_slot.transform(mdc.mat3(_width / 9, _depth / 9, 1))
        _rectangular_trough_slot = _rectangular_trough_slot.transform(mdc.vec3(_position_x, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _rectangular_trough_slot)

        return updated_model

    def rectangular_blind_slot(self):
        _rectangular_blind_slot = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Rectangular_Blind_Slot.stl')

        _rectangular_blind_slot.mergeclose()
        _rectangular_blind_slot = mdc.segmentation(_rectangular_blind_slot)

        _length = np.random.uniform(1, self.max_scale)
        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)
        _position_x = np.random.uniform(0.5, (9.5 - _width))

        _rectangular_blind_slot = _rectangular_blind_slot.transform(mdc.mat3(_width / 9, _length / 9, _depth / 9))
        _rectangular_blind_slot = _rectangular_blind_slot.transform(mdc.vec3(_position_x, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _rectangular_blind_slot)

        return updated_model

    def triangular_pocket(self):
        _triangular_pocket = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Triangular_Pocket.stl')

        _triangular_pocket.mergeclose()
        _triangular_pocket = mdc.segmentation(_triangular_pocket)

        _side_length = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _side_length)
        _position_y = np.random.uniform(0.5, 9.5 - _side_length)

        _triangular_pocket = _triangular_pocket.transform(mdc.mat3(_side_length / 9, _side_length / 9, _depth / 9))
        _triangular_pocket = _triangular_pocket.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _triangular_pocket)

        return updated_model

    def rectangular_pocket(self):
        _rectangular_pocket = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Rectangular_Pocket.stl')

        _rectangular_pocket.mergeclose()
        _rectangular_pocket = mdc.segmentation(_rectangular_pocket)

        _length = np.random.uniform(1, self.max_scale)
        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _length)
        _position_y = np.random.uniform(0.5, 9.5 - _width)

        _rectangular_pocket = _rectangular_pocket.transform(mdc.mat3(_length / 9, _width / 9, _depth / 9))
        _rectangular_pocket = _rectangular_pocket.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _rectangular_pocket)

        return updated_model

    def circular_end_pocket(self):
        _circular_end_pocket = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Circular_End_Pocket.stl')

        _circular_end_pocket.mergeclose()
        _circular_end_pocket = mdc.segmentation(_circular_end_pocket)

        _length = np.random.uniform(1, self.max_scale)
        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _length)
        _position_y = np.random.uniform(0.5, 9.5 - _width)

        _circular_end_pocket = _circular_end_pocket.transform(mdc.mat3(_length / 9, _width / 4.5, _depth / 9))
        _circular_end_pocket = _circular_end_pocket.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _circular_end_pocket)

        return updated_model

    def triangular_blind_step(self):
        _triangular_blind_step = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Triangular_Blind_Step.stl')

        _triangular_blind_step.mergeclose()
        _triangular_blind_step = mdc.segmentation(_triangular_blind_step)

        _side_length = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _triangular_blind_step = _triangular_blind_step.transform(
            mdc.mat3(_side_length / 9, _depth / 9, _side_length / 9))
        _triangular_blind_step = _triangular_blind_step.transform(mdc.vec3(-0.01, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _triangular_blind_step)

        return updated_model

    def circular_blind_step(self):
        _circular_blind_step = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Circular_Blind_Step.stl')

        _circular_blind_step.mergeclose()
        _circular_blind_step = mdc.segmentation(_circular_blind_step)

        _radius = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _circular_blind_step = _circular_blind_step.transform(mdc.mat3(_radius / 9, _depth / 9, _radius / 9))
        _circular_blind_step = _circular_blind_step.transform(mdc.vec3(-0.01, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _circular_blind_step)

        return updated_model

    def rectangular_blind_step(self):
        _rectangular_blind_step = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Rectangular_Blind_Step.stl')

        _rectangular_blind_step.mergeclose()
        _rectangular_blind_step = mdc.segmentation(_rectangular_blind_step)

        _length = np.random.uniform(1, self.max_scale)
        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _rectangular_blind_step = _rectangular_blind_step.transform(mdc.mat3(_length / 9, _depth / 9, _width / 9))
        _rectangular_blind_step = _rectangular_blind_step.transform(mdc.vec3(-0.01, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _rectangular_blind_step)

        return updated_model

    def rectangular_trough_step(self):
        _rectangular_trough_step = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Rectangular_Trough_Step.stl')

        _rectangular_trough_step.mergeclose()
        _rectangular_trough_step = mdc.segmentation(_rectangular_trough_step)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _rectangular_trough_step = _rectangular_trough_step.transform(mdc.mat3(1, _depth / 9, _width / 9))
        _rectangular_trough_step = _rectangular_trough_step.transform(mdc.vec3(-0.25, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _rectangular_trough_step)

        return updated_model

    def two_side_through_step(self):
        _two_side_through_step = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Two_Side_Trough_Step.stl')

        _two_side_through_step.mergeclose()
        _two_side_through_step = mdc.segmentation(_two_side_through_step)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _two_side_through_step = _two_side_through_step.transform(mdc.mat3(1, _depth / 9, _width / 9))
        _two_side_through_step = _two_side_through_step.transform(mdc.vec3(-0.25, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _two_side_through_step)

        return updated_model

    def slanted_through_step(self):
        _slanted_through_step = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Slanted_Trough_Step.stl')

        _slanted_through_step.mergeclose()
        _slanted_through_step = mdc.segmentation(_slanted_through_step)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _slanted_through_step = _slanted_through_step.transform(mdc.mat3(1, _depth / 9, _width / 9))
        _slanted_through_step = _slanted_through_step.transform(mdc.vec3(-0.25, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _slanted_through_step)

        return updated_model

    def chamfer(self):
        _chamfer = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Chamfer.stl')

        _chamfer.mergeclose()
        _chamfer = mdc.segmentation(_chamfer)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _chamfer = _chamfer.transform(mdc.mat3((_width / 9), 1.1, _depth / 9))
        _chamfer = _chamfer.transform(mdc.vec3(-0.01, -0.1, -0.01))

        updated_model = mdc.difference(self.model, _chamfer)

        return updated_model

    def round(self):
        _round = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Round.stl')

        _round.mergeclose()
        _round = mdc.segmentation(_round)

        _width = np.random.uniform(1, self.max_scale)
        _depth = np.random.uniform(1, self.max_depth)

        _round = _round.transform(mdc.mat3((_width / 9), 1.1, _depth / 9))
        _round = _round.transform(mdc.vec3(-0.01, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _round)

        return updated_model

    def vertical_circular_end_blind_slot(self):
        _vertical_circular_end_blind_slot = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Vertical_Circular_End_Blind_Slot.stl')

        _vertical_circular_end_blind_slot.mergeclose()
        _vertical_circular_end_blind_slot = mdc.segmentation(_vertical_circular_end_blind_slot)

        _length = np.random.uniform(self.min_scale, self.max_scale)
        _width = np.random.uniform(self.min_scale, self.max_scale)
        _depth = np.random.uniform(self.min_depth, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _width)

        _vertical_circular_end_blind_slot = _vertical_circular_end_blind_slot.transform(
            mdc.mat3((_width / 9), (_length / 9), _depth / 9))
        _vertical_circular_end_blind_slot = _vertical_circular_end_blind_slot.transform(
            mdc.vec3(_position_x, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _vertical_circular_end_blind_slot)

        return updated_model

    def horizontal_circular_end_blind_slot(self):
        _horizontal_circular_end_blind_slot = mdc.read(
            os.getenv('TEMPLATES_SOURCE') + '/Horizontal_Circular_End_Blind_Slot.stl')

        _horizontal_circular_end_blind_slot.mergeclose()
        _horizontal_circular_end_blind_slot = mdc.segmentation(_horizontal_circular_end_blind_slot)

        _length = np.random.uniform(self.min_scale, self.max_scale)
        _width = np.random.uniform(self.min_scale, self.max_scale)
        _depth = np.random.uniform(self.min_depth, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _width)

        _horizontal_circular_end_blind_slot = _horizontal_circular_end_blind_slot.transform(
            mdc.mat3((_width / 9), (_length / 9), _depth / 9))
        _horizontal_circular_end_blind_slot = _horizontal_circular_end_blind_slot.transform(
            mdc.vec3(_position_x, -0.01, -0.01))

        updated_model = mdc.difference(self.model, _horizontal_circular_end_blind_slot)

        return updated_model

    def six_side_passage(self):
        _six_side_pocket = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Six_Side_Passage.stl')

        _six_side_pocket.mergeclose()
        _six_side_pocket = mdc.segmentation(_six_side_pocket)

        _diameter = np.random.uniform(self.min_scale, self.max_scale)
        _position_x = np.random.uniform(0.5, 9.5 - _diameter)
        _position_y = np.random.uniform(0.5, 9.5 - _diameter)

        _six_side_pocket = _six_side_pocket.transform(mdc.mat3((_diameter / 9), (_diameter / 9), 1))
        _six_side_pocket = _six_side_pocket.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _six_side_pocket)

        return updated_model

    def six_side_pocket(self):
        _six_side_pocket = mdc.read(os.getenv('TEMPLATES_SOURCE') + '/Six_Side_Pocket.stl')

        _six_side_pocket.mergeclose()
        _six_side_pocket = mdc.segmentation(_six_side_pocket)

        _diameter = np.random.uniform(self.min_scale, self.max_scale)
        _depth = np.random.uniform(self.max_depth, self.max_depth)
        _position_x = np.random.uniform(0.5, 9.5 - _diameter)
        _position_y = np.random.uniform(0.5, 9.5 - _diameter)

        _six_side_pocket = _six_side_pocket.transform(mdc.mat3((_diameter / 9), (_diameter / 9), (_depth / 9)))
        _six_side_pocket = _six_side_pocket.transform(mdc.vec3(_position_x, _position_y, -0.01))

        updated_model = mdc.difference(self.model, _six_side_pocket)

        return updated_model
