import math
import numpy as np
import numpy.random
import madcad as mdc


class MachiningFeature:
    def __init__(self, machining_feature):
        self.machining_feature = machining_feature

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
        _cube = mdc.brick(width=mdc.vec3(10))
        _cube = _cube.transform(mdc.vec3(5, 5, 5))

        _outside_ring_radius = np.random.uniform(1, 4.5)
        _inside_ring_radius = np.random.uniform(_outside_ring_radius / 3, _outside_ring_radius - 0.2)
        _position_x = np.random.uniform(_outside_ring_radius + 0.5, 9.5 - _outside_ring_radius)
        _position_y = np.random.uniform(_outside_ring_radius + 0.5, 9.5 - _outside_ring_radius)
        _depth = np.random.uniform(1, 9)

        outside_ring = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, _depth), mdc.vec3(_position_x, _position_y, 10.01), _outside_ring_radius)
        inside_ring = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, _depth), mdc.vec3(_position_x, _position_y, 10.01), _inside_ring_radius)
        o_ring = mdc.difference(outside_ring, inside_ring)

        updated_model = mdc.difference(_cube, o_ring)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)
        mdc.show([updated_model])
        return updated_model

    def trough_hole(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _radius = np.random.uniform(0.5, 4.5)
        _position_x = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _position_y = np.random.uniform(_radius + 0.5, 9.5 - _radius)

        cylinder = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, 0), mdc.vec3(_position_x, _position_y, 10), _radius)

        updated_model = mdc.difference(cube, cylinder)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def blind_hole(self):

        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _radius = np.random.uniform(0.5, 4.5)
        _position_x = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _position_y = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _depth = np.random.uniform(1, 9)

        cylinder = mdc.cylinder(
            mdc.vec3(_position_x, _position_y, _depth), mdc.vec3(_position_x, _position_y, 10), _radius)

        updated_model = mdc.difference(cube, cylinder)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def triangular_passage(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        L = np.random.uniform(1, 9)
        D = 10.02
        X = np.random.uniform(0.5, 9.5 - L)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.01)
        B = mdc.vec3(L, 0, 10.01)
        C = mdc.vec3(L / 2, L * math.sin(math.radians(60)), 10.01)

        triangular_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        triangular_passage = mdc.extrusion(-D * mdc.Z, mdc.flatsurface(triangular_passage))
        triangular_passage = triangular_passage.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        updated_model = mdc.difference(cube, triangular_passage)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_passage(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        L = np.random.uniform(1, 9)
        W = np.random.uniform(1, 9)
        Depth = 10.02
        X = np.random.uniform(0.5, 9.5 - W)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.01)
        B = mdc.vec3(W, 0, 10.01)
        C = mdc.vec3(W, L, 10.01)
        D = mdc.vec3(0, L, 10.01)

        _rectangular_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_passage = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_passage))
        _rectangular_passage = _rectangular_passage.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        updated_model = mdc.difference(cube, _rectangular_passage)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def circular_trough_slot(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        R = np.random.uniform(1, 4.5)
        X = np.random.uniform(R + 0.5, 9.5 - R)

        _circular_trough_slot = mdc.cylinder(mdc.vec3(X, -0.01, 10), mdc.vec3(X, 10.01, 10), R)

        updated_model = mdc.difference(cube, _circular_trough_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def triangular_trough_slot(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        W = np.random.uniform(1, 9)
        D = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - W)
        A = mdc.vec3(0, -0.01, 10.01)
        B = mdc.vec3(W, -0.01, 10.01)
        C = mdc.vec3(W / 2, -0.01, 10.01 - D)
        _lines = [mdc.Segment(B, A), mdc.Segment(A, C), mdc.Segment(C, B)]
        _triangular_trough_slot_web = mdc.web(_lines)
        _triangular_trough_slot = mdc.extrusion(10.02 * mdc.Y, mdc.flatsurface(_triangular_trough_slot_web))
        _triangular_trough_slot = _triangular_trough_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))
        updated_model = mdc.difference(cube, _triangular_trough_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_trough_slot(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        W = np.random.uniform(1, 9)
        D = np.random.uniform(1, 9)
        H = 10.02
        X = np.random.uniform(0.5, 9.5 - W)
        A = mdc.vec3(0, -0.01, 10.01)
        B = mdc.vec3(W, -0.01, 10.01)
        C = mdc.vec3(W, -0.01, 10.01 - D)
        D = mdc.vec3(0, -0.01, 10.01 - D)

        _rectangular_trough_slot = [mdc.Segment(B, A), mdc.Segment(C, B), mdc.Segment(D, C), mdc.Segment(A, D)]
        _rectangular_trough_slot = mdc.extrusion(H * mdc.Y, mdc.flatsurface(_rectangular_trough_slot))
        _rectangular_trough_slot = _rectangular_trough_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        updated_model = mdc.difference(cube, _rectangular_trough_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_blind_slot(self):
        cube = mdc.brick(width=mdc.vec3(10, 10, 10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        W = np.random.uniform(1, 9)
        D = np.random.uniform(1, 9)
        H = np.random.uniform(1, 9)

        Y = np.random.uniform(0.5, 9.5 - W)

        A = mdc.vec3(10.01, 0, 10.01)
        B = mdc.vec3(10.01, 0, 10.01 - D)
        C = mdc.vec3(10.01, W, 10.01 - D)
        D = mdc.vec3(10.01, W, 10.01)

        _rectangular_blind_slot = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_blind_slot = mdc.extrusion(-H * mdc.X, mdc.flatsurface(_rectangular_blind_slot))
        _rectangular_blind_slot = _rectangular_blind_slot.transform(mdc.translate(mdc.vec3(0, Y, 0)))

        updated_model = mdc.difference(cube, _rectangular_blind_slot)
        updated_model = mdc.segmentation(updated_model)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def triangular_pocket(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        L = np.random.uniform(1, 9)
        D = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - L)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.01)
        B = mdc.vec3(L, 0, 10.01)
        C = mdc.vec3(L / 2, L * math.sin(math.radians(60)), 10.01)

        _triangular_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _triangular_pocket = mdc.extrusion(-D * mdc.Z, mdc.flatsurface(_triangular_pocket))
        _triangular_pocket = _triangular_pocket.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        updated_model = mdc.difference(cube, _triangular_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_pocket(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        L = np.random.uniform(1, 9)
        W = np.random.uniform(1, 9)
        Depth = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - W)
        Y = np.random.uniform(0.5, 9.5 - L)
        A = mdc.vec3(0, 0, 10.01)
        B = mdc.vec3(W, 0, 10.01)
        C = mdc.vec3(W, L, 10.01)
        D = mdc.vec3(0, L, 10.01)

        _rectangular_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_pocket = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_pocket))
        _rectangular_pocket = _rectangular_pocket.transform(mdc.translate(mdc.vec3(X, Y, 0)))
        updated_model = mdc.difference(cube, _rectangular_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def circular_end_pocket(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _width = np.random.uniform(1, 8)
        _length = np.random.uniform(1, 9 - _width)
        _depth = 10.02
        X = np.random.uniform(0.5, 9 - _width)
        Y = np.random.uniform(0.5, 9.5 - (_width + _length))

        A = mdc.vec3(0, (_width / 2), 10.01)
        B = mdc.vec3((_width / 2), 0, 10.01)
        C = mdc.vec3(_width, (_width / 2), 10.01)
        D = mdc.vec3(_width, (_length + (_width / 2)), 10.01)
        E = mdc.vec3((_width / 2), (_length + _width), 10.01)
        F = mdc.vec3(0, (_length + (_width / 2)), 10.01)

        _circular_end_pocket = [mdc.ArcThrough(A, B, C), mdc.Segment(C, D), mdc.ArcThrough(D, E, F), mdc.Segment(F, A)]
        _circular_end_pocket = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_circular_end_pocket))
        _circular_end_pocket = _circular_end_pocket.transform(mdc.translate(mdc.vec3(X, Y, 0)))

        updated_model = mdc.difference(cube, _circular_end_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def triangular_blind_step(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        X = np.random.uniform(1, 9)
        Y = np.random.uniform(1, 9)
        _depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.01, 10.01, 10.01)
        B = mdc.vec3(X, 10.01, 10.01)
        C = mdc.vec3(10.01, Y, 10.01)

        _triangular_blind_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _triangular_blind_step = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_triangular_blind_step))

        updated_model = mdc.difference(cube, _triangular_blind_step)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def circular_blind_step(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        R = np.random.uniform(1, 9)
        _depth = np.random.uniform(1, 9)

        _circular_blind_step = mdc.cylinder(mdc.vec3(-0.01, -0.01, 10.01), mdc.vec3(-0.01, -0.01, _depth), R)

        updated_model = mdc.difference(cube, _circular_blind_step)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_blind_step(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        L = np.random.uniform(1, 9)
        W = np.random.uniform(1, 9)
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.01, 10.01, 10.01)
        B = mdc.vec3(L, 10.01, 10.01)
        C = mdc.vec3(L, W, 10.01)
        D = mdc.vec3(10.01, W, 10.01)

        _rectangular_blind_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_blind_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_blind_step))

        updated_model = mdc.difference(cube, _rectangular_blind_step)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def rectangular_trough_step(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        W = np.random.uniform(1, 9)
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.01, 10.01, 10.01)
        B = mdc.vec3(W, 10.01, 10.01)
        C = mdc.vec3(W, -0.01, 10.01)
        D = mdc.vec3(10.01, -0.01, 10.01)

        _rectangular_blind_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _rectangular_blind_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_rectangular_blind_step))

        updated_model = mdc.difference(cube, _rectangular_blind_step)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def two_side_through_step(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _width_A = np.random.uniform(1, 8)
        _width_B = np.random.uniform(_width_A + 0.5, 9)
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.01, 10.01, 10.01)
        B = mdc.vec3(-0.01, 10.01, 10.01)
        C = mdc.vec3(-0.01, 10.01 - _width_A, 10.01)
        D = mdc.vec3(5, 10.01 - _width_B, 10.01)
        E = mdc.vec3(10.01, 10.01 - _width_A, 10.01)

        _two_side_through_step = [
            mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E), mdc.Segment(E, A)]
        _two_side_through_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_two_side_through_step))

        updated_model = mdc.difference(cube, _two_side_through_step)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def slanted_through_step(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _width_A = np.random.uniform(1, 9)
        _width_B = np.random.uniform(1, 9)
        Depth = np.random.uniform(1, 9)

        A = mdc.vec3(10.01, 10.01, 10.01)
        B = mdc.vec3(-0.01, 10.01, 10.01)
        C = mdc.vec3(-0.01, 10.01 - _width_A, 10.01)
        D = mdc.vec3(10.01, 10.01 - _width_B, 10.01)

        _slanted_through_step = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, A)]
        _slanted_through_step = mdc.extrusion(-Depth * mdc.Z, mdc.flatsurface(_slanted_through_step))

        updated_model = mdc.difference(cube, _slanted_through_step)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def chamfer(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        X = np.random.uniform(1, 9)
        Z = np.random.uniform(1, 9)
        Depth = 10.02

        A = mdc.vec3(10.01, -0.01, 10.01)
        B = mdc.vec3(X, -0.01, 10.01)
        C = mdc.vec3(10.01, -0.01, Z)

        _chamfer = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, A)]
        _chamfer = mdc.extrusion(Depth * mdc.Y, mdc.flatsurface(_chamfer))

        updated_model = mdc.difference(cube, _chamfer)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def round(self):
        print("hello")
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))
        R = np.random.uniform(1, 9)
        Z = R - (R * math.sin(math.radians(45)))
        Y = R - (R * math.sin(math.radians(45)))
        _depth = 10.02

        A = mdc.vec3(-0.01, 10.01 - R, 10.01)
        B = mdc.vec3(-0.01, 10.01, 10.01)
        C = mdc.vec3(-0.01, 10.01, 10.01 - R)
        D = mdc.vec3(-0.01, 10.01 - Y, 10.01 - Z)

        _round = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.ArcThrough(C, D, A)]
        _round = mdc.extrusion(_depth * mdc.X, mdc.flatsurface(_round))

        updated_model = mdc.difference(cube, _round)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def vertical_circular_end_blind_slot(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _length = np.random.uniform(1, 9)
        _width = np.random.uniform(1, 9 - _length)
        _depth = np.random.uniform(1, 9)
        X = np.random.uniform(0.5, 9.5 - _length)

        A = mdc.vec3(0, 10.01, 10.01)
        B = mdc.vec3(0, (10.01 - _width), 10.01)
        C = mdc.vec3(_length / 2, 10.01 - (_width + (_length / 2)), 10.01)
        D = mdc.vec3(_length, (10.01 - _width), 10.01)
        E = mdc.vec3(_length, 10.01, 10.01)

        _v_circular_end_blind_slot = [mdc.Segment(A, B), mdc.ArcThrough(B, C, D), mdc.Segment(D, E), mdc.Segment(E, A)]
        _v_circular_end_blind_slot = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_v_circular_end_blind_slot))
        _v_circular_end_blind_slot = _v_circular_end_blind_slot.transform(mdc.translate(mdc.vec3(X, 0, 0)))

        updated_model = mdc.difference(cube, _v_circular_end_blind_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def horizontal_circular_end_blind_slot(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _width = np.random.uniform(1, 4)
        _length = np.random.uniform(1, 9 - (2 * _width))
        _radius = (_width * math.sin(math.radians(45)))
        _depth = np.random.uniform(1, 9)

        A = mdc.vec3(_width, 5, 10.01)
        B = mdc.vec3(_width, (5 + (_length / 2)), 10.01)
        C = mdc.vec3(_radius, (5 + (_length / 2) + _radius), 10.01)
        D = mdc.vec3(-0.01, (5 + _width + (_length / 2)), 10.01)
        E = mdc.vec3(-0.01, (5 - (_width + (_length / 2))), 10.01)
        F = mdc.vec3(_radius, (5 - (_radius + (_length / 2))), 10.01)
        G = mdc.vec3(_width, (5 - (_length / 2)), 10.01)

        _h_circular_end_blind_slot = \
            [mdc.Segment(A, B), mdc.ArcThrough(B, C, D), mdc.Segment(D, E), mdc.ArcThrough(E, F, G), mdc.Segment(G, A)]
        _h_circular_end_blind_slot = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_h_circular_end_blind_slot))

        updated_model = mdc.difference(cube, _h_circular_end_blind_slot)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def six_side_passage(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _radius = np.random.uniform(1, 4.5)
        _Cx = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _Cy = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _depth = 10.02

        A = mdc.vec3(_radius, 0, 10.01)
        B = mdc.vec3(_radius * math.cos(math.radians(-300)), _radius * math.sin(math.radians(-300)), 10.01)
        C = mdc.vec3(_radius * math.cos(math.radians(-240)), _radius * math.sin(math.radians(-240)), 10.01)
        D = mdc.vec3(- _radius, 0, 10.01)
        E = mdc.vec3(_radius * math.cos(math.radians(-120)), _radius * math.sin(math.radians(-120)), 10.01)
        F = mdc.vec3(_radius * math.cos(math.radians(-60)), _radius * math.sin(math.radians(-60)), 10.01)

        _six_side_passage = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E),
                             mdc.Segment(E, F), mdc.Segment(F, A)]
        _six_side_passage = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_six_side_passage))
        _six_side_passage = _six_side_passage.transform(mdc.translate(mdc.vec3(_Cx, _Cy, 0)))

        updated_model = mdc.difference(cube, _six_side_passage)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model

    def six_side_pocket(self):
        cube = mdc.brick(width=mdc.vec3(10))
        cube = cube.transform(mdc.vec3(5, 5, 5))

        _radius = np.random.uniform(1, 4.5)
        _Cx = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _Cy = np.random.uniform(_radius + 0.5, 9.5 - _radius)
        _depth = np.random.uniform(1, 9)

        A = mdc.vec3(_radius, 0, 10.01)
        B = mdc.vec3(_radius * math.cos(math.radians(-300)), _radius * math.sin(math.radians(-300)), 10.01)
        C = mdc.vec3(_radius * math.cos(math.radians(-240)), _radius * math.sin(math.radians(-240)), 10.01)
        D = mdc.vec3(- _radius, 0, 10.01)
        E = mdc.vec3(_radius * math.cos(math.radians(-120)), _radius * math.sin(math.radians(-120)), 10.01)
        F = mdc.vec3(_radius * math.cos(math.radians(-60)), _radius * math.sin(math.radians(-60)), 10.01)

        _six_side_pocket = [mdc.Segment(A, B), mdc.Segment(B, C), mdc.Segment(C, D), mdc.Segment(D, E),
                            mdc.Segment(E, F), mdc.Segment(F, A)]
        _six_side_pocket = mdc.extrusion(-_depth * mdc.Z, mdc.flatsurface(_six_side_pocket))

        _six_side_pocket = _six_side_pocket.transform(mdc.translate(mdc.vec3(_Cx, _Cy, 0)))

        updated_model = mdc.difference(cube, _six_side_pocket)
        updated_model.mergeclose()
        updated_model = mdc.segmentation(updated_model)

        return updated_model
