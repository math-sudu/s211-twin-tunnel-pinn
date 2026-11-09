"""Changing excavation boundaries and initial-stress release above a tunnel.

z=0 is the final pit base; the original surface is z=H.  Incremental soil
displacements are measured from equilibrium before excavation.  The retained
domain carries -sigma_initial n on newly exposed faces.  The prescribed
initial effective stress is diag(K0, K0, 1) * -gamma_prime * (H-z).
"""

import numpy as np

from .ground import excavation_pressures, excavation_rectangles, treatment_mask


def excavate_mesh(mesh, config, depths_m):
    """Remove complete conforming wedges, retaining the remaining cover.

    The parent mesh must contain the floor levels and every block edge.  All
    stages use its nodes and cells before unused vertices are compacted.
    """
    names = list(config['excavation_blocks_m'])
    excavation_pressures(config, names, depths_m)
    vertices = mesh.p[:, mesh.t]
    centres = vertices.mean(axis=1)
    removed = np.zeros(mesh.nelements, dtype=bool)
    protected = treatment_mask(centres, config.get('retained_wall_boxes_m', []))
    volume = 0.
    for name, (a, b, c, d) in zip(names, excavation_rectangles(config, names)):
        floor = config['surface_height_m'] - depths_m[name]
        selected = ((centres[0] > a) & (centres[0] < b) &
                    (centres[1] > c) & (centres[1] < d) & (centres[2] > floor))
        if np.any(selected):
            low, high = vertices[:, :, selected].min(axis=1), vertices[:, :, selected].max(axis=1)
            if (np.any(low < np.array([a, c, floor])[:, None] - 1e-8) or
                    np.any(high[:2] > np.array([b, d])[:, None] + 1e-8)):
                raise ValueError('Parent mesh does not conform to the excavation block and floor.')
        # An unselected element must not cross the floor from below either.
        horizontal = ((centres[0] > a) & (centres[0] < b) &
                      (centres[1] > c) & (centres[1] < d))
        if np.any(horizontal & (vertices[2].min(axis=0) < floor - 1e-8) &
                  (vertices[2].max(axis=0) > floor + 1e-8)):
            raise ValueError('Parent mesh does not conform to the excavation floor.')
        removed |= selected & ~protected
        volume += (b - a) * (d - c) * depths_m[name]
    if np.any(protected):
        p = vertices[:, :, removed]
        cross = np.cross((p[:, 1]-p[:, 0]).T, (p[:, 2]-p[:, 0]).T)
        volume = np.sum(np.linalg.norm(cross, axis=1)/2 * abs(p[0, 3]-p[0, 0]))
    return mesh.remove_elements(np.flatnonzero(removed)), float(volume)


def excavation_surface_load(basis, config, lateral_stress_ratio):
    """Integrate initial-stress release on floors and vertical pit walls.

    Triangle degree-two and rectangular two-by-two rules integrate linear
    pressure times linear/bilinear nodal shapes, including its first moments.
    Ground-surface, remote and pre-existing tunnel faces receive no new load.
    """
    if not np.isfinite(lateral_stress_ratio) or lateral_stress_ratio < 0:
        raise ValueError('The lateral stress ratio must be finite and nonnegative.')
    mesh, H = basis.mesh, config['surface_height_m']
    gamma = config['effective_unit_weight_kN_m3']
    force = np.zeros(basis.N)
    directions = np.array([lateral_stress_ratio, lateral_stress_ratio, 1.])
    for face in mesh.boundary_facets():
        nodes = np.unique(mesh.facets[:, face])
        p = mesh.p[:, nodes]
        mid = p.mean(axis=1)
        if (mid[2] < -1e-8 or np.isclose(mid[2], H) or
                np.all(np.isclose(abs(p[0]), config['half_width_m'])) or
                np.all(np.isclose(abs(p[1]), config['cross_half_width_m']))):
            continue
        normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        norm = np.linalg.norm(normal)
        normal /= norm
        element = mesh.f2t[0, face]
        centre = mesh.p[:, mesh.t[:, element]].mean(axis=1)
        if normal @ (mid - centre) < 0:
            normal *= -1
        if len(nodes) == 3:
            shape = np.array([[2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]])
            weights = np.full(3, norm / 6)
        else:
            axes = np.flatnonzero(np.ptp(p, axis=1) > 1e-8)
            if len(axes) != 2:
                raise ValueError('Excavation quadrilateral faces must be axis aligned.')
            q = np.array([-1., 1.]) / np.sqrt(3)
            qa, qb = np.meshgrid(q, q)
            signs = 2 * (p[axes] - p[axes].min(axis=1)[:, None]) / np.ptp(p[axes], axis=1)[:, None] - 1
            shape = ((1 + qa.ravel()[:, None] * signs[0]) *
                     (1 + qb.ravel()[:, None] * signs[1]) / 4)
            weights = np.full(4, np.prod(np.ptp(p[axes], axis=1)) / 4)
        z = shape @ p[2]
        nodal_pressure = shape.T @ (weights * gamma * (H - z))
        force[basis.nodal_dofs[:, nodes]] += (directions * normal)[:, None] * nodal_pressure
    return force
