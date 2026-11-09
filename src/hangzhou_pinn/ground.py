"""Finite-footprint, three-dimensional elastic free-field input below the pit."""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import splu
from skfem import Basis, FacetBasis, BilinearForm, LinearForm
from skfem import MeshHex, ElementHex1, ElementVector, asm
from skfem.helpers import ddot, sym_grad, trace


def unloading_modulus(compression_modulus, nu, multiplier):
    return compression_modulus * (1 + nu) * (1 - 2 * nu) / (1 - nu) * multiplier


def graded_axis(core, extent, spacing, anchors=()):
    """Resolve the loaded/treatment zone and grade the distant ground outward."""
    positive = list(np.linspace(0, core, int(np.ceil(core / spacing)) + 1))
    step = spacing
    while positive[-1] < extent:
        step *= 1.5
        positive.append(min(positive[-1] + step, extent))
    return np.unique(np.r_[-np.asarray(positive)[:0:-1], positive, anchors])


def excavation_rectangles(config, blocks):
    """Resolve named load rectangles in tunnel/road coordinates, in metres."""
    pit = config['excavation_width_m'] / 2
    cross = config['excavation_cross_length_m'] / 2
    rectangles = config.get('excavation_blocks_m', {
        'left': [-pit, 0, -cross, cross],
        'right': [0, pit, -cross, cross],
    })
    if len(set(blocks)) != len(blocks):
        raise ValueError('An excavation block may occur only once in a stage.')
    selected = [rectangles[name] for name in blocks]
    for a, b, c, d in selected:
        if not (-pit <= a < b <= pit and -cross <= c < d <= cross):
            raise ValueError('Excavation rectangles must lie inside the configured footprint.')
    for i, (a, b, c, d) in enumerate(selected):
        for e, f, g, h in selected[:i]:
            if max(a, e) < min(b, f) and max(c, g) < min(d, h):
                raise ValueError('Excavation blocks must not overlap.')
    return selected


def excavation_pressures(config, blocks, block_depths_m=None):
    """Release pressures for explicit removed depths on the fixed base plane.

    The depth describes removed overburden, not a moving mesh surface. Existing
    complete-block callers use the configured final depth. Explicit stage maps
    must name every selected block, including zero-depth blocks.
    """
    full_depth = config['excavation_depth_m']
    if not np.isfinite(full_depth) or full_depth <= 0:
        raise ValueError('The configured excavation depth must be finite and positive.')
    if block_depths_m is None:
        depths = np.full(len(blocks), full_depth)
    else:
        if set(block_depths_m) != set(blocks):
            raise ValueError('Explicit depths must name exactly the selected excavation blocks.')
        depths = np.array([block_depths_m[name] for name in blocks], dtype=float)
    if np.any(~np.isfinite(depths)) or np.any((depths < 0) | (depths > full_depth)):
        raise ValueError('Block depths must lie between zero and the configured final depth.')
    return config['effective_unit_weight_kN_m3'] * depths


def reinforcement_boxes(config):
    """Treatment volumes as [xmin, xmax, ymin, ymax, zmin, zmax], in metres."""
    pit = config['excavation_width_m'] / 2
    cross = config['excavation_cross_length_m'] / 2
    boxes = config.get('reinforcement_boxes_m', [
        [-pit, pit, -cross, cross, -config['reinforcement_thickness_m'], 0.0],
    ])
    for i, (a, b, c, d, e, f) in enumerate(boxes):
        if not (-config['half_width_m'] < a < b < config['half_width_m'] and
                -config['cross_half_width_m'] < c < d < config['cross_half_width_m'] and
                -config['depth_below_pit_base_m'] < e < f <= 0):
            raise ValueError('Treatment boxes must lie inside the ground domain, away from remote faces.')
        for u, v, w, x, y, z in boxes[:i]:
            if max(a, u) < min(b, v) and max(c, w) < min(d, x) and max(e, y) < min(f, z):
                raise ValueError('Treatment boxes must not overlap.')
    return boxes


def treatment_mask(points, boxes):
    mask = np.zeros_like(points[0], dtype=bool)
    for a, b, c, d, e, f in boxes:
        mask |= ((points[0] > a) & (points[0] < b) &
                 (points[1] > c) & (points[1] < d) &
                 (points[2] > e) & (points[2] < f))
    return mask


def halfspace_boundary(points, config, blocks, young, order=16, block_depths_m=None):
    """Integrate Boussinesq displacement on remote boundary nodes.

    x follows the tunnel, y the road; z is upward from the pit-base plane.
    The finite rectangle carries upward release pressure. This is a boundary
    condition for the free-field generator, not a tunnel-solution comparator.
    """
    pressures = excavation_pressures(config, blocks, block_depths_m)
    nu = config['poisson_ratio']
    mu = young / (2 * (1 + nu))
    nodes, weights = leggauss(order)
    sources, loads = [], []
    for (a, b, c, d), pressure in zip(excavation_rectangles(config, blocks), pressures):
        sx, sy = np.meshgrid((a + b) / 2 + nodes * (b - a) / 2,
                             (c + d) / 2 + nodes * (d - c) / 2)
        sources.append(np.c_[sx.ravel(), sy.ravel()])
        loads.append((np.outer(weights, weights) * (b - a) * (d - c) / 4 * pressure).ravel())
    if not sources:
        return np.zeros_like(points)
    source, load = np.concatenate(sources), np.concatenate(loads)
    displacement = np.empty_like(points)
    for start in range(0, points.shape[1], 256):
        p = points[:, start:start + 256]
        dx = p[0, :, None] - source[None, :, 0]
        dy = p[1, :, None] - source[None, :, 1]
        depth = -p[2, :, None]
        distance = np.sqrt(dx**2 + dy**2 + depth**2)
        radial = (1 - 2 * nu) / (distance * (distance + depth)) - depth / distance**3
        factor = load / (4 * np.pi * mu)
        displacement[0, start:start + 256] = (dx * radial) @ factor
        displacement[1, start:start + 256] = (dy * radial) @ factor
        displacement[2, start:start + 256] = (depth**2 / distance**3 + 2 * (1 - nu) / distance) @ factor
    return displacement


class GroundModel:
    """Reuse one stiffness factorization across the stage load vectors."""

    def __init__(self, config, reinforced=False, mesh_factor=1.0):
        self.config = config
        pit = config['excavation_width_m'] / 2
        cross = config['excavation_cross_length_m'] / 2
        dx, dy, dz = [config[key] / mesh_factor for key in ['mesh_dx_m', 'mesh_dy_m', 'mesh_dz_m']]
        names = list(config.get('excavation_blocks_m', {'left': None, 'right': None}))
        rectangles = excavation_rectangles(config, names)
        boxes = reinforcement_boxes(config) if reinforced else []
        xs = graded_axis(36, config['half_width_m'], dx,
                         [-pit, pit] + [v for r in rectangles for v in r[:2]] +
                         [v for box in boxes for v in box[:2]])
        ys = graded_axis(36, config['cross_half_width_m'], dy,
                         [-cross, cross] + [v for r in rectangles for v in r[2:]] +
                         [v for box in boxes for v in box[2:4]])
        zs = graded_axis(12, config['depth_below_pit_base_m'], dz,
                         [-config['axis_below_base_m'], -config['reinforcement_thickness_m']] +
                         [v for box in boxes for v in box[4:]])
        zs = zs[zs <= 0]
        self.mesh = mesh = MeshHex.init_tensor(xs, ys, zs)
        element = ElementVector(ElementHex1())
        self.basis = basis = Basis(mesh, element, intorder=3)
        self.reinforcement_boxes = boxes
        self.reinforcement_volume = float(np.sum(treatment_mask(basis.global_coordinates(), boxes) * basis.dx))
        self.young = young = unloading_modulus(config['compression_modulus_kPa'],
                                               config['poisson_ratio'], config['unloading_multiplier'])
        nu = config['poisson_ratio']

        @BilinearForm
        def stiffness(u, v, w):
            E = young * np.where(treatment_mask(w.x, boxes), config['reinforcement_multiplier'], 1.0)
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)

        self.K = asm(stiffness, basis)
        remote = (np.isclose(np.abs(mesh.p[0]), config['half_width_m']) |
                  np.isclose(np.abs(mesh.p[1]), config['cross_half_width_m']) |
                  np.isclose(mesh.p[2], -config['depth_below_pit_base_m']))
        self.remote_nodes = np.flatnonzero(remote)
        self.fixed = basis.nodal_dofs[:, remote].ravel()
        self.free = np.setdiff1d(np.arange(basis.N), self.fixed)
        print(f'Ground mesh: {mesh.nvertices} nodes, {mesh.nelements} elements; factorizing.', flush=True)
        self.factor = splu(self.K[self.free][:, self.free].tocsc())
        top = mesh.facets_satisfying(lambda x: np.isclose(x[2], 0))
        self.top_basis = FacetBasis(mesh, element, facets=top, intorder=3)
        sample_half_width = config.get('axis_sampling_half_width_m', 110.)
        if sample_half_width > config['half_width_m']:
            raise ValueError('Axis sampling must remain inside the ground mesh.')
        self.x = np.linspace(-sample_half_width, sample_half_width,
                             int(np.ceil(2 * sample_half_width / .25)) + 1)
        scalar_basis = basis.split_bases()[2]
        self.probe = scalar_basis.probes(np.vstack([self.x, np.zeros_like(self.x),
                                                   np.full_like(self.x, -config['axis_below_base_m'])]))
        self.native_x = np.unique(np.r_[self.x[0], xs[(xs > self.x[0]) & (xs < self.x[-1])], self.x[-1]])
        self.native_probe = scalar_basis.probes(np.vstack([
            self.native_x, np.zeros_like(self.native_x),
            np.full_like(self.native_x, -config['axis_below_base_m'])]))

    def solve_stage(self, blocks, block_depths_m=None):
        c, basis, mesh = self.config, self.basis, self.mesh
        rectangles = excavation_rectangles(c, blocks)
        pressures = excavation_pressures(c, blocks, block_depths_m)

        @LinearForm
        def release(v, w):
            traction = np.zeros_like(w.x[0])
            for (a, b, c, d), pressure in zip(rectangles, pressures):
                traction += pressure * ((w.x[0] >= a) & (w.x[0] <= b) &
                                        (w.x[1] >= c) & (w.x[1] <= d))
            return traction * v[2]

        f = asm(release, self.top_basis)
        u = np.zeros(basis.N)
        prescribed = halfspace_boundary(mesh.p[:, self.remote_nodes], c, blocks, self.young,
                                       block_depths_m=block_depths_m)
        u[self.fixed] = prescribed.ravel()
        rhs = (f - self.K @ u)[self.free]
        u[self.free] = self.factor.solve(rhs)
        reaction = self.K @ u - f
        z_dofs = basis.nodal_dofs[2]
        g = self.probe @ u[z_dofs]
        return {'x_m': self.x, 'g_m': np.asarray(g),
                'native_x_m': self.native_x,
                'native_g_m': np.asarray(self.native_probe @ u[z_dofs]),
                'mesh_nodes': int(mesh.nvertices), 'mesh_elements': int(mesh.nelements),
                'load_kN': float(f[z_dofs].sum()), 'reaction_kN': float(reaction[z_dofs].sum()),
                'elastic_energy_twice_kN_m': float(u @ (self.K @ u)),
                'applied_work_kN_m': float(u @ f), 'boundary_work_kN_m': float(u @ reaction),
                'young_kPa': float(self.young),
                'reinforcement_boxes_m': self.reinforcement_boxes,
                'reinforcement_volume_m3': self.reinforcement_volume,
                'boundary': 'Finite rectangular half-space displacement on remote faces; treatment perturbation truncated there.'}


def beam_and_foundation(config):
    t, g, f = config['tunnel'], config['ground'], config['foundation']
    d, di = t['outer_diameter_m'], t['inner_diameter_m']
    area = np.pi * (d**2 - di**2) / 4
    inertia = np.pi * (d**4 - di**4) / 64
    B = t['concrete_modulus_kPa'] * inertia * t['bending_reduction']
    S = (t['concrete_modulus_kPa'] / (2 * (1 + t['poisson_ratio'])) * area *
         t['shear_reduction'] * t['shear_correction'])
    E = unloading_modulus(g['compression_modulus_kPa'], g['poisson_ratio'], g['unloading_multiplier'])
    ku, kl = E * d / f['upper_thickness_m'], E * d / f['lower_thickness_m']
    T = E / (2 * (1 + g['poisson_ratio'])) * d * f['shear_layer_thickness_m']
    return {'B': B, 'S': S, 'ku': ku, 'kl': kl, 'T': T}
