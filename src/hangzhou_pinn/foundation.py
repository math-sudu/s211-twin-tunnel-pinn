"""Work-conjugate ground restraint for vertical motion of a circular tunnel.

The soil cavity has prescribed normal motion and free tangential slip.  A
Schur complement eliminates every unconstrained soil displacement.  The
result maps longitudinal displacement coefficients [m] to forces [kN].
"""

import gmsh
import numpy as np
from time import perf_counter
from scipy.linalg import solve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from skfem import Basis, BilinearForm, ElementVector, ElementWedge1, MeshWedge1, asm
from skfem.helpers import ddot, sym_grad, trace
from skfem.quadrature import get_quadrature
from skfem.refdom import RefLine, RefTri

from .ground import (excavation_pressures, excavation_rectangles, graded_axis,
                     treatment_mask, unloading_modulus)


class _PardisoFactor:
    """Optional sparse factor with the same RHS interface as SuperLU."""
    def __init__(self, matrix):
        from pypardiso import PyPardisoSolver
        self.matrix = matrix.tocsr()
        self.solver = PyPardisoSolver()
        self.solver.factorize(self.matrix)

    def solve(self, rhs):
        if rhs.ndim == 2 and rhs.shape[1] == 0:
            return np.empty_like(rhs)
        return self.solver.solve(self.matrix, rhs)

    def close(self):
        self.solver.free_memory(everything=True)


def excavation_load(basis, config, blocks, block_depths_m=None):
    """Integrate release pressure over the extruded mesh's rectangular top faces.

    Exact bilinear shape integrals also resolve a block edge inside a face.
    The four node weights preserve total force and both surface first moments.
    """
    return horizontal_surface_load(basis, excavation_rectangles(config, blocks),
                                   excavation_pressures(config, blocks, block_depths_m))


def horizontal_surface_load(basis, rectangles, pressures, height=0.):
    """Integrate vertical pressure on horizontal boundary faces at a given level."""
    mesh = basis.mesh
    boundary = mesh.boundary_facets()
    faces = mesh.facets[:, boundary[np.all(np.isclose(mesh.p[2, mesh.facets[:, boundary]], height), axis=0)]]
    x, y = mesh.p[0, faces], mesh.p[1, faces]
    xmin, xmax, ymin, ymax = x.min(axis=0), x.max(axis=0), y.min(axis=0), y.max(axis=0)
    force = np.zeros(basis.N)
    for (a, b, c, d), pressure in zip(rectangles, pressures):
        left, right = np.maximum(xmin, a), np.minimum(xmax, b)
        low, high = np.maximum(ymin, c), np.minimum(ymax, d)
        sel = (right > left) & (high > low)
        ix1 = ((right[sel] - xmin[sel])**2 - (left[sel] - xmin[sel])**2) / (2 * (xmax[sel] - xmin[sel]))
        iy1 = ((high[sel] - ymin[sel])**2 - (low[sel] - ymin[sel])**2) / (2 * (ymax[sel] - ymin[sel]))
        ix = np.where(np.isclose(x[:, sel], xmin[sel]), right[sel] - left[sel] - ix1, ix1)
        iy = np.where(np.isclose(y[:, sel], ymin[sel]), high[sel] - low[sel] - iy1, iy1)
        np.add.at(force, basis.nodal_dofs[2, faces[:, sel]].ravel(), (pressure * ix * iy).ravel())
    return force


def wedge_quadrature():
    """Triangle (x,y) times interval z, exact for first-order wedge energies.

    Explicit tensor construction follows ElementWedge1's reference coordinates.
    scikit-fem 12.0.1's default RefWedge quadrature instead has weight sum 1/4
    and incorrectly mixes the triangle and interval coordinates.
    """
    triangle, triangle_weights = get_quadrature(RefTri, 2)
    line, line_weights = get_quadrature(RefLine, 3)
    points = np.vstack([np.repeat(triangle, len(line_weights), axis=1),
                        np.tile(line, len(triangle_weights))])
    weights = np.outer(triangle_weights, line_weights).ravel()
    return points, weights


def cavity_mesh(config, radius, mesh_factor=1.0):
    """Extrude a conforming circular-hole cross-section in the tunnel direction.

    Gmsh's two planar coordinates become physical y,z.  The cap rectangle
    resolves every study layout's common transverse and vertical boundary;
    x stations resolve their longitudinal faces.  Wedges preserve x reflection.
    A positive surface_height_m retains cover above the final base.  Its
    excavation_levels_m and block edges partition a common mesh for later cuts.
    """
    c = config
    cross, depth = c['cross_half_width_m'], c['depth_below_pit_base_m']
    axis, roof = c['axis_below_base_m'], c['reinforcement_thickness_m']
    surface = c.get('surface_height_m', 0.)
    if not (radius < axis < depth - radius and radius < cross):
        raise ValueError('The circular cavity must be strictly below the surface and inside the domain.')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('relative_motion_cross_section')
        occ = gmsh.model.occ
        rectangle = occ.addRectangle(-cross, -depth, 0, 2 * cross, depth + surface)
        centres = c.get('cavity_centres_y_m', [0.])
        holes = [(2, occ.addDisk(y, -axis, 0, radius, radius)) for y in centres]
        soil, _ = occ.cut([(2, rectangle)], holes)
        half_roof = c['excavation_cross_length_m'] / 2
        cap = occ.addRectangle(-half_roof, -roof, 0, 2 * half_roof, roof)
        partitions = [(2, cap)]
        if surface > 0:
            rectangles = excavation_rectangles(c, list(c['excavation_blocks_m']))
            ys = np.unique([v for box in rectangles for v in box[2:]])
            levels = np.unique([0., surface, *c['excavation_levels_m']])
            if levels[0] < 0 or levels[-1] > surface:
                raise ValueError('Excavation levels must lie in the original cover.')
            for left, right in zip(ys[:-1], ys[1:]):
                for low, high in zip(levels[:-1], levels[1:]):
                    partitions.append((2, occ.addRectangle(left, low, 0, right - left, high - low)))
        for left, right, low, high in c.get('cross_section_partitions_m', []):
            partitions.append((2, occ.addRectangle(left, low, 0, right - left, high - low)))
        occ.fragment(soil, partitions)
        occ.synchronize()

        def size(dim, tag, y, z, unused, lc):
            gap = max(min(np.hypot(y-centre, z+axis) for centre in centres)-radius,0.)
            roof_gap = np.hypot(max(abs(y) - half_roof, 0), max(-roof - z, z, 0))
            return min(c['far_mesh_size_m'], c['cavity_mesh_size_m'] + 0.45 * gap,
                       c['roof_mesh_size_m'] + 0.45 * roof_gap) / mesh_factor

        gmsh.model.mesh.setSizeCallback(size)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)
        _, triangle_tags = gmsh.model.mesh.getElementsByType(2)
        lookup = np.empty(int(np.max(tags)) + 1, dtype=int)
        lookup[tags] = np.arange(len(tags))
        triangles = lookup[np.asarray(triangle_tags).reshape(-1, 3)]
        # OCC construction points that do not belong to triangles carry no DOFs.
        used, inverse = np.unique(triangles, return_inverse=True)
        cross_points = coordinates[used, :2]
        triangles = inverse.reshape(-1, 3)
    finally:
        gmsh.finalize()

    anchors = list(c['longitudinal_anchors_m'])
    if surface > 0:
        anchors.extend(v for box in rectangles for v in box[:2])
    xs = graded_axis(36, c['half_width_m'], c['longitudinal_mesh_size_m'] / mesh_factor, anchors)
    n = len(cross_points)
    points = np.vstack([np.repeat(xs, n), np.tile(cross_points.T, len(xs))])
    cells = np.vstack([np.c_[triangles + i * n, triangles + (i + 1) * n]
                       for i in range(len(xs) - 1)]).T
    return MeshWedge1(points, cells)


def motion_basis(x, stations):
    """Linear shape functions, zero at and beyond their outer support stations."""
    x, stations = np.asarray(x), np.asarray(stations)
    if np.any(np.diff(stations) <= 0):
        raise ValueError('Displacement stations must be strictly increasing.')
    i = np.searchsorted(stations, x).clip(1, len(stations) - 1)
    t = (x - stations[i - 1]) / (stations[i] - stations[i - 1])
    values = np.zeros((len(x), len(stations)))
    values[np.arange(len(x)), i - 1] = 1 - t
    values[np.arange(len(x)), i] = t
    values[(x <= stations[0]) | (x >= stations[-1])] = 0
    return values[:, 1:-1]


def cavity_constraints(basis, config, radius, stations):
    """u = Z v + A delta: free slip v, prescribed normal motion n_z*delta.

    At cavity nodes (n_y,n_z), (u_y,u_z) = (-n_z,n_y)*v_t
    + (n_y,n_z)*n_z*delta(x).  The x displacement is free.  Every component
    vanishes on remote faces; the ground surface carries natural zero traction.
    """
    mesh, dofs = basis.mesh, basis.nodal_dofs
    x, y, z = mesh.p
    centres = np.asarray(config.get('cavity_centres_y_m', [0.]))
    nearest = np.argmin(abs(y[:, None] - centres), axis=1)
    local_y = y - centres[nearest]
    radial = np.hypot(local_y, z + config['axis_below_base_m'])
    remote = (np.isclose(abs(x), config['half_width_m']) |
              np.isclose(abs(y), config['cross_half_width_m']) |
              np.isclose(z, -config['depth_below_pit_base_m']))
    cavity = np.isclose(radial, radius, atol=1e-8, rtol=0) & ~remote
    ordinary = ~remote & ~cavity
    ids = dofs[:, ordinary].ravel()
    axial = dofs[0, cavity]
    nc, no, na = np.sum(cavity), len(ids), len(axial)
    ny, nz = local_y[cavity] / radius, (z[cavity] + config['axis_below_base_m']) / radius
    rows = np.r_[ids, axial, dofs[1, cavity], dofs[2, cavity]]
    cols = np.r_[np.arange(no + na), no + na + np.arange(nc), no + na + np.arange(nc)]
    values = np.r_[np.ones(no + na), -nz, ny]
    Z = coo_matrix((values, (rows, cols)), shape=(basis.N, no + na + nc)).tocsr()
    shape = motion_basis(x[cavity], stations)
    n = len(stations) - 2
    translations = n * len(centres)
    ovalization = config.get('cavity_ovalization', False)
    A = np.zeros((basis.N, translations * (2 if ovalization else 1)))
    for j in range(len(centres)):
        selected = nearest[cavity] == j
        A[dofs[1, cavity][selected], j*n:(j+1)*n] = (ny*nz)[selected, None] * shape[selected]
        A[dofs[2, cavity][selected], j*n:(j+1)*n] = nz[selected, None]**2 * shape[selected]
        if ovalization:
            # q*cos(2 theta) is an inextensional ring's radial ovalization.
            # Tangential soil motion remains free at the slipping interface.
            angular = (ny**2-nz**2)[selected, None]
            columns = slice(translations+j*n, translations+(j+1)*n)
            A[dofs[1, cavity][selected], columns] = ny[selected, None]*angular*shape[selected]
            A[dofs[2, cavity][selected], columns] = nz[selected, None]*angular*shape[selected]
    return Z, A, cavity, remote


class CavityFoundation:
    """Condense the same cavity geometry for each material layout."""

    def __init__(self, mesh, config, radius, stations):
        self.config, self.stations = config, np.asarray(stations)
        self.basis = Basis(mesh, ElementVector(ElementWedge1()), quadrature=wedge_quadrature())
        self.Z, self.A, self.cavity, self.remote = cavity_constraints(
            self.basis, config, radius, stations)
        self.radius = radius

    def condense(self, boxes, contrast, *, additional_regions=(), formed_components=(),
                 replacement_regions=(), load_only=False):
        """Condense bonded elastic regions on the retained soil mesh.

        Additional regions specify a name, boxes_m and modulus_multiplier.
        They share the soil Poisson ratio and the pre-excavation reference.
        Excavated elements contribute neither stiffness nor region volume.
        Formed components add stiffness and their fixed connection-reference
        loads before elimination. These are separate from soil region changes.
        """
        c, basis, Z, A = self.config, self.basis, self.Z, self.A
        young = unloading_modulus(c['compression_modulus_kPa'], c['poisson_ratio'], c['unloading_multiplier'])
        nu = c['poisson_ratio']
        regions = [{'name': 'base_treatment', 'boxes_m': boxes,
                    'modulus_multiplier': contrast}, *additional_regions]
        seen_boxes = []
        names = set()
        for region in regions:
            if region['name'] in names:
                raise ValueError('Material region names must be unique.')
            names.add(region['name'])
            multiplier = region['modulus_multiplier']
            if not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError('Material modulus multipliers must be finite and positive.')
            for box in region['boxes_m']:
                x0, x1, y0, y1, z0, z1 = box
                if not (np.all(np.isfinite(box)) and
                        -c['half_width_m'] < x0 < x1 < c['half_width_m'] and
                        -c['cross_half_width_m'] < y0 < y1 < c['cross_half_width_m'] and
                        -c['depth_below_pit_base_m'] < z0 < z1 <= c.get('surface_height_m', 0.)):
                    raise ValueError('Material boxes must lie inside the soil domain.')
                low, high = np.asarray(box)[::2], np.asarray(box)[1::2]
                for previous in seen_boxes:
                    if np.all(np.maximum(low, previous[::2]) < np.minimum(high, previous[1::2])):
                        raise ValueError('Material boxes must not overlap.')
                seen_boxes.append(np.asarray(box))

        coordinates = basis.global_coordinates()
        multipliers = np.ones_like(basis.dx)
        material_volumes = {}
        for region in regions:
            mask = treatment_mask(coordinates, region['boxes_m'])
            multipliers[mask] = region['modulus_multiplier']
            material_volumes[region['name']] = float(np.sum(mask * basis.dx))

        # Concrete walls replace soil/treatment within their retained volume.
        for region in replacement_regions:
            mask = treatment_mask(coordinates, region['boxes_m'])
            multipliers[mask] = region['modulus_multiplier']
            material_volumes[region['name']] = float(np.sum(mask * basis.dx))

        @BilinearForm
        def stiffness(u, v, w):
            E = young * multipliers
            mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
            eu, ev = sym_grad(u), sym_grad(v)
            return 2 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)

        K = asm(stiffness, basis)
        reference_force = np.zeros(basis.N)
        for component in formed_components:
            if component.compatibility.shape[1] != basis.N:
                raise ValueError('A formed component must use the current soil displacement space.')
            added_stiffness, added_force = component.assembled_terms()
            K = K + added_stiffness
            reference_force += added_force
        print(f'Cavity restraint: {basis.mesh.nvertices} nodes, {basis.mesh.nelements} wedges; factorizing.', flush=True)
        Kfree = (Z.T @ K @ Z).tocsc()
        self.release_factor()
        factor_started = perf_counter()
        factor = (_PardisoFactor(Kfree) if c.get('linear_solver') == 'pardiso' else splu(Kfree))
        factor_seconds = perf_counter() - factor_started
        self.K, self.factor, self.reference_force = K, factor, reference_force
        if load_only:
            self.U = None
            return None, None, {'factorization_seconds':factor_seconds}
        print(f'Factorized in {factor_seconds:.1f} s; recovering cavity modes.', flush=True)
        recovery_started = perf_counter()
        KA = K @ A
        U = A - Z @ factor.solve(np.asarray(Z.T @ KA))
        reactions = K @ U
        operator = A.T @ reactions
        work = U.T @ reactions
        scale = np.linalg.norm(operator)
        checks = {
            'mesh_nodes': int(basis.mesh.nvertices), 'mesh_elements': int(basis.mesh.nelements),
            'free_dofs': int(Z.shape[1]), 'cavity_nodes': int(np.sum(self.cavity)),
            'factorization_seconds': factor_seconds,
            'mode_recovery_seconds': perf_counter() - recovery_started,
            'reinforcement_volume_m3': material_volumes['base_treatment'],
            'material_region_volumes_m3': material_volumes,
            'cavity_volume_m3': float(2 * c['half_width_m'] * 2 * c['cross_half_width_m'] *
                                      (c['depth_below_pit_base_m'] + c.get('surface_height_m', 0.)) -
                                      c.get('removed_volume_m3', 0.) - np.sum(basis.dx)),
            'circular_cavity_volume_m3': float(2 * c['half_width_m'] * np.pi * self.radius**2 *
                                              len(c.get('cavity_centres_y_m', [0.]))),
            'reciprocity_relative_gap': float(np.linalg.norm(operator - operator.T) / scale),
            'work_relative_gap': float(np.linalg.norm(operator - work) / scale),
            'free_equilibrium_relative_gap': float(np.linalg.norm(Z.T @ reactions) / np.linalg.norm(reactions)),
            'reflection_relative_gap': float(np.linalg.norm(operator - operator[::-1, ::-1]) / scale),
            'smallest_eigenvalue_kN_m': float(np.linalg.eigvalsh((operator + operator.T) / 2)[0]),
        }
        self.K, self.factor, self.U = K, factor, U
        self.reference_force = reference_force
        return operator, U, checks

    def condense_loading(self, force, lift):
        """Reduce excavation work with u = lift + Z v + A d.

        lift sets remote displacement and a reference cavity normal motion.
        After eliminating v, potential = d^T C d/2 - h^T d + constant.
        Here b=force+reference_force includes formed-component references.
        The independent work expression U^T (b-K lift) checks h's sign and
        includes the effect of nonzero remote displacement.
        """
        total_force = force + self.reference_force
        effective = total_force - self.K @ lift
        u0 = lift + self.Z @ self.factor.solve(np.asarray(self.Z.T @ effective))
        residual = self.K @ u0 - total_force
        load = -self.A.T @ residual
        checks = {
            'free_equilibrium_relative_gap': float(np.linalg.norm(self.Z.T @ residual) / max(np.linalg.norm(total_force), 1e-12)),
            'applied_vertical_force_kN': float(force[self.basis.nodal_dofs[2]].sum()),
            'vertical_balance_kN': float((residual + total_force)[self.basis.nodal_dofs[2]].sum()),
        }
        if self.U is not None:
            checks['virtual_work_relative_gap'] = float(np.linalg.norm(load-self.U.T@effective)/max(np.linalg.norm(load),1e-12))
        return load, u0, checks

    def recover_displacement(self, coefficients, particular_displacement):
        """Recover u0 + U a, including internal connection displacements.

        particular_displacement is u0 returned by condense_loading for the
        current force, lift and formed components, not a stress-free state.
        coefficients use the same cavity basis and reference as that lift.
        """
        return np.asarray(particular_displacement) + self.U @ coefficients

    def release_factor(self):
        """Release external sparse storage before assembling another stage."""
        factor = getattr(self, 'factor', None)
        if isinstance(factor, _PardisoFactor):
            factor.close()
        self.factor = None


def kerr_operator(stations, ku, kl, shear):
    """Condense the Kerr internal layer on a 1D linear trial space.

    Auxiliary layer flux is zero at both ends; prescribed relative tunnel
    motion is zero at the outer support stations.  This is a constitutive
    reduction check, not an alternative tunnel solver.
    """
    grid = np.unique(np.concatenate([np.linspace(a, b, int(np.ceil((b - a) / 1.5)) + 1)
                                     for a, b in zip(stations[:-1], stations[1:])]))
    n = len(grid)
    mass, gradient = np.zeros((n, n)), np.zeros((n, n))
    for i, h in enumerate(np.diff(grid)):
        mass[i:i + 2, i:i + 2] += h / 6 * np.array([[2., 1.], [1., 2.]])
        gradient[i:i + 2, i:i + 2] += np.array([[1., -1.], [-1., 1.]]) / h
    shape = motion_basis(grid, stations)
    coupling = ku * mass @ shape
    return shape.T @ coupling - coupling.T @ solve((ku + kl) * mass + shear * gradient, coupling, assume_a='pos')
