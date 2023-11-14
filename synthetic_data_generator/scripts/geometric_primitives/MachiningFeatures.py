import math
import numpy as np
import numpy.random
import madcad as mdc


# The following class follows the provided data by zhan et al.
# https://github.com/madlabub/Machining-feature-dataset/tree/master
# please note that the provides machining feature description ("FeatureList.pdf") differs for some cad models.
# Therefore, the following methods may differ from the machining feature description and follows more the real provided
# cad models for better comparison with the current state of the art machining feature recognition algorithms like
# MsvNet and SsdNet.
# For the names of the vectors we followed however the provided machining feature description for better
# understanding. However, this may not follow the PEP8 guidelines


class MachiningFeatures:
    def __init__(self, machining_feature_id, max_machining_feature_dimension):
        self.machining_feature_id = machining_feature_id
        self.max_machining_feature_dimension = max_machining_feature_dimension

        self.machining_feature = [self.o_ring, self.trough_hole,
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

    def generate(self):
        return self.machining_feature[self.machining_feature_id]()

    def o_ring(self):
        _outside_ring_radius = np.random.uniform(1, (4.5 * self.max_machining_feature_dimension))
        _inside_ring_radius = np.random.uniform(_outside_ring_radius / 3, _outside_ring_radius - 0.2)
        _position_x = np.random.uniform(_outside_ring_radius + 0.5, 9.5 - _outside_ring_radius)
        _position_y = np.random.uniform(_outside_ring_radius + 0.5, 9.5 - _outside_ring_radius)
        _depth = np.random.uniform(1, 9)

        outside_ring = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, _depth), mdc.vec3(_position_x, _position_y, 10.2), _outside_ring_radius)
        inside_ring = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, _depth), mdc.vec3(_position_x, _position_y, 10.2), _inside_ring_radius)
        o_ring = mdc.difference(outside_ring, inside_ring)

        return o_ring

    def trough_hole(self):
        _radius = np.random.uniform(0.5, (4.5 * self.max_machining_feature_dimension))
        _position_x = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _position_y = np.random.uniform(_radius + 0.5, 9.5 - _radius)

        trough_hole = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, -0.2), mdc.vec3(_position_x, _position_y, 10.2), _radius)

        return trough_hole

    def blind_hole(self):
        _radius = np.random.uniform(0.5, (4.5 * self.max_machining_feature_dimension))
        _position_x = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _position_y = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _depth = np.random.uniform(1, 9)

        blind_hole = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, _depth), mdc.vec3(_position_x, _position_y, 10.2), _radius)

        return blind_hole

    def triangular_passage(self):
        L = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        D = 10.4
        X = np.random.uniform(0.5, 9.5 - L)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.2)
        B = mdc.vec3(L, 0, 10.2)
        C = mdc.vec3(L / 2, L * math.sin(math.radians(60)), 10.02)

        triangular_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        triangular_passage = mdc.extrusion(-D * mdc.Z, mdc.flatsurface(triangular_passage))
        triangular_passage = triangular_passage.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        return triangular_passage

    def rectangular_passage(self):
        L = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Depth = 10.4
        X = np.random.uniform(0.5, 9.5 - W)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.2)
        B = mdc.vec3(W, 0, 10.2)
        C = mdc.vec3(W, L, 10.2)
        D = mdc.vec3(0, L, 10.2)

        _rectangular_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_passage = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_passage))
        _rectangular_passage = _rectangular_passage.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        return _rectangular_passage

    def circular_trough_slot(self):
        R = np.random.uniform(1, (4.5 * self.max_machining_feature_dimension))
        X = np.random.uniform(R + 0.5, 9.5 - R)

        _circular_trough_slot = mdc.cylinder(mdc.vec3(X, -0.2, 10), mdc.vec3(X, 10.2, 10), R)

        return _circular_trough_slot

    def triangular_trough_slot(self):
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        D = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        X = np.random.uniform(0.5, 9.5 - W)
        A = mdc.vec3(0, -0.2, 10.2)
        B = mdc.vec3(W, -0.2, 10.2)
        C = mdc.vec3(W / 2, -0.2, 10.2 - D)
        _lines = [mdc.Segment(B, A), mdc.Segment(A, C), mdc.Segment(C, B)]
        _triangular_trough_slot_web = mdc.web(_lines)
        _triangular_trough_slot = mdc.extrusion(10.4 * mdc.Y, mdc.flatsurface(_triangular_trough_slot_web))
        _triangular_trough_slot = _triangular_trough_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        return _triangular_trough_slot

    def rectangular_trough_slot(self):
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        D = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        H = 10.4
        X = np.random.uniform(0.5, 9.5 - W)
        A = mdc.vec3(-0.2, -0.2, 10.2)
        B = mdc.vec3(W, -0.2, 10.2)
        C = mdc.vec3(W, -0.2, 10.2 - D)
        D = mdc.vec3(-0.2, -0.2, 10.2 - D)

        _rectangular_trough_slot = [mdc.Segment(B, A), mdc.Segment(C, B), mdc.Segment(D, C), mdc.Segment(A, D)]
        _rectangular_trough_slot = mdc.extrusion(H * mdc.Y, mdc.flatsurface(_rectangular_trough_slot))
        _rectangular_trough_slot = _rectangular_trough_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        return _rectangular_trough_slot

    def rectangular_blind_slot(self):
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        D = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        H = np.random.uniform(1, 9)

        Y = np.random.uniform(0.5, 9.5 - W)

        A = mdc.vec3(10.2, 0, 10.2)
        B = mdc.vec3(10.2, 0, 10.2 - D)
        C = mdc.vec3(10.2, W, 10.2 - D)
        D = mdc.vec3(10.2, W, 10.2)

        _rectangular_blind_slot = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_blind_slot = mdc.extrusion(-H * mdc.X, mdc.flatsurface(_rectangular_blind_slot))
        _rectangular_blind_slot = _rectangular_blind_slot.transform(mdc.translate(mdc.vec3(0, Y, 0)))

        return _rectangular_blind_slot

    def triangular_pocket(self):
        L = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        D = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - L)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.2)
        B = mdc.vec3(L, 0, 10.2)
        C = mdc.vec3(L / 2, L * math.sin(math.radians(60)), 10.2)

        _triangular_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _triangular_pocket = mdc.extrusion(-D * mdc.Z, mdc.flatsurface(_triangular_pocket))
        _triangular_pocket = _triangular_pocket.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        return _triangular_pocket

    def rectangular_pocket(self):
        L = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Depth = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - W)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.2)
        B = mdc.vec3(W, 0, 10.2)
        C = mdc.vec3(W, L, 10.2)
        D = mdc.vec3(0, L, 10.2)

        _rectangular_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_pocket = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_pocket))
        _rectangular_pocket = _rectangular_pocket.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        return _rectangular_pocket

    def circular_end_pocket(self):
        _width = np.random.uniform(1, (8 * self.max_machining_feature_dimension))
        _length = np.random.uniform(1, (9 * self.max_machining_feature_dimension) - _width)
        _depth = 10.4
        X = np.random.uniform(0.5, 9 - _width)
        Y = np.random.uniform(0.5, 9.5 - (_width + _length))

        A = mdc.vec3(0, (_width / 2), 10.2)
        B = mdc.vec3((_width / 2), 0, 10.2)
        C = mdc.vec3(_width, (_width / 2), 10.2)
        D = mdc.vec3(_width, (_length + (_width / 2)), 10.2)
        E = mdc.vec3((_width / 2), (_length + _width), 10.2)
        F = mdc.vec3(0, (_length + (_width / 2)), 10.2)

        _circular_end_pocket = [mdc.ArcThrough(A, B, C), mdc.Segment(C, D), mdc.ArcThrough(D, E, F), mdc.Segment(F, A)]
        _circular_end_pocket = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_circular_end_pocket))
        _circular_end_pocket = _circular_end_pocket.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        return _circular_end_pocket

    def triangular_blind_step(self):
        X = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Y = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        _depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.2, 10.2, 10.2)
        B = mdc.vec3(X, 10.2, 10.2)
        C = mdc.vec3(10.2, Y, 10.2)

        _triangular_blind_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _triangular_blind_step = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_triangular_blind_step))

        return _triangular_blind_step

    def circular_blind_step(self):
        R = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        _depth = np.random.uniform(1, 9)

        _circular_blind_step = mdc.cylinder(mdc.vec3(-0.2, -0.2, 10.2), mdc.vec3(-0.2, -0.2, _depth), R)

        return _circular_blind_step

    def rectangular_blind_step(self):
        L = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.2, 10.2, 10.2)
        B = mdc.vec3(L, 10.2, 10.2)
        C = mdc.vec3(L, W, 10.2)
        D = mdc.vec3(10.2, W, 10.2)

        _rectangular_blind_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_blind_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_blind_step))

        return _rectangular_blind_step

    def rectangular_trough_step(self):
        W = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.2, 10.2, 10.2)
        B = mdc.vec3(W, 10.2, 10.2)
        C = mdc.vec3(W, -0.2, 10.2)
        D = mdc.vec3(10.2, -0.2, 10.2)

        _rectangular_blind_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_blind_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_blind_step))

        return _rectangular_blind_step

    def two_side_through_step(self):
        _width_A = np.random.uniform(1, (8 * self.max_machining_feature_dimension))
        _width_B = np.random.uniform(_width_A + 0.5, (9 * self.max_machining_feature_dimension))
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.2, 10.2, 10.2)
        B = mdc.vec3(-0.2, 10.2, 10.2)
        C = mdc.vec3(-0.2, 10.2 - _width_A, 10.2)
        D = mdc.vec3(5, 10.2 - _width_B, 10.2)
        E = mdc.vec3(10.2, 10.2 - _width_A, 10.2)

        _two_side_through_step = [
            mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E), mdc.Segment(E, A)]
        _two_side_through_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_two_side_through_step))

        return _two_side_through_step

    def slanted_through_step(self):
        _width_A = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        _width_B = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.2, 10.2, 10.2)
        B = mdc.vec3(-0.2, 10.2, 10.2)
        C = mdc.vec3(-0.2, 10.2 - _width_A, 10.2)
        D = mdc.vec3(10.2, 10.2 - _width_B, 10.2)

        _slanted_through_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _slanted_through_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_slanted_through_step))

        return _slanted_through_step

    def chamfer(self):
        X = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Z = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Depth = 10.4

        A = mdc.vec3(10.2, -0.2, 10.2)
        B = mdc.vec3(X, -0.2, 10.2)
        C = mdc.vec3(10.2, -0.2, Z)

        _chamfer = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _chamfer = mdc.extrusion(Depth * mdc.Y, mdc.flatsurface(_chamfer))

        return _chamfer

    def round(self):
        R = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        Z = R - (R * math.sin(math.radians(45)))
        Y = R - (R * math.sin(math.radians(45)))
        _depth = 10.4

        A = mdc.vec3(-0.2, 10.2 - R, 10.2)
        B = mdc.vec3(-0.2, 10.2, 10.2)
        C = mdc.vec3(-0.2, 10.2, 10.2 - R)
        D = mdc.vec3(-0.2, 10.2 - Y, 10.2 - Z)

        _round = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.ArcThrough(C, D, A)]
        _round = mdc.extrusion(_depth * mdc.X, mdc.flatsurface(_round))

        return _round

    def vertical_circular_end_blind_slot(self):
        _length = np.random.uniform(1, (9 * self.max_machining_feature_dimension))
        _width = np.random.uniform(1, (9 * self.max_machining_feature_dimension) - _length)
        _depth = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - _length)

        A = mdc.vec3(0, 10.2, 10.2)
        B = mdc.vec3(0, (10.2 - _width), 10.2)
        C = mdc.vec3(_length / 2, 10.2 - (_width + (_length / 2)), 10.2)
        D = mdc.vec3(_length, (10.2 - _width), 10.2)
        E = mdc.vec3(_length, 10.2, 10.2)

        _v_circular_end_blind_slot = [mdc.Segment(A, B), mdc.ArcThrough(B, C, D), mdc.Segment(D, E), mdc.Segment(E, A)]
        _v_circular_end_blind_slot = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_v_circular_end_blind_slot))
        _v_circular_end_blind_slot = _v_circular_end_blind_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        return _v_circular_end_blind_slot

    def horizontal_circular_end_blind_slot(self):
        _width = np.random.uniform(1, (4 * self.max_machining_feature_dimension))
        _length = np.random.uniform(1, (9 * self.max_machining_feature_dimension) - (2 * _width))
        _radius = (_width * math.sin(math.radians(45)))
        _depth = np.random.uniform(1, 9)

        A = mdc.vec3(_width, 5, 10.2)
        B = mdc.vec3(_width, (5 + (_length / 2)), 10.2)
        C = mdc.vec3(_radius, (5 + (_length / 2) + _radius), 10.2)
        D = mdc.vec3(-0.2, (5 + _width + (_length / 2)), 10.2)
        E = mdc.vec3(-0.2, (5 - (_width + (_length / 2))), 10.2)
        F = mdc.vec3(_radius, (5 - (_radius + (_length / 2))), 10.2)
        G = mdc.vec3(_width, (5 - (_length / 2)), 10.2)

        _h_circular_end_blind_slot = \
            [mdc.Segment(A, B), mdc.ArcThrough(B, C, D), mdc.Segment(D, E), mdc.ArcThrough(E, F, G), mdc.Segment(G, A)]
        _h_circular_end_blind_slot = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_h_circular_end_blind_slot))

        return _h_circular_end_blind_slot

    def six_side_passage(self):
        _radius = np.random.uniform(1, (4.5 * self.max_machining_feature_dimension))
        _Cx = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _Cy = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _depth = 10.4

        A = mdc.vec3(_radius, 0, 10.2)
        B = mdc.vec3(_radius * math.cos(math.radians(-300)), _radius * math.sin(math.radians(-300)), 10.2)
        C = mdc.vec3(_radius * math.cos(math.radians(-240)), _radius * math.sin(math.radians(-240)), 10.2)
        D = mdc.vec3(- _radius, 0, 10.2)
        E = mdc.vec3(_radius * math.cos(math.radians(-120)), _radius * math.sin(math.radians(-120)), 10.2)
        F = mdc.vec3(_radius * math.cos(math.radians(-60)), _radius * math.sin(math.radians(-60)), 10.2)

        _six_side_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E),
                             mdc.Segment(E, F), mdc.Segment(F, A)]
        _six_side_passage = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_six_side_passage))
        _six_side_passage = _six_side_passage.transform(mdc.translate(mdc.vec3(_Cx, _Cy, 0)))

        return _six_side_passage

    def six_side_pocket(self):
        _radius = np.random.uniform(1, (4.5 * self.max_machining_feature_dimension))
        _Cx = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _Cy = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _depth = np.random.uniform(1, 9)

        A = mdc.vec3(_radius, 0, 10.2)
        B = mdc.vec3(_radius * math.cos(math.radians(-300)), _radius * math.sin(math.radians(-300)), 10.2)
        C = mdc.vec3(_radius * math.cos(math.radians(-240)), _radius * math.sin(math.radians(-240)), 10.2)
        D = mdc.vec3(- _radius, 0, 10.2)
        E = mdc.vec3(_radius * math.cos(math.radians(-120)), _radius * math.sin(math.radians(-120)), 10.2)
        F = mdc.vec3(_radius * math.cos(math.radians(-60)), _radius * math.sin(math.radians(-60)), 10.2)

        _six_side_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E),
                            mdc.Segment(E, F), mdc.Segment(F, A)]
        _six_side_pocket = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_six_side_pocket))
        _six_side_pocket = _six_side_pocket.transform(mdc.translate(mdc.vec3(_Cx, _Cy, 0)))

        return _six_side_pocket
