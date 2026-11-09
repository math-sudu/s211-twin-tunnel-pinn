"""Four-field mixed PINN with a work-conjugate condensed soil operator.

The continuum cavity replaces the two auxiliary Kerr-layer states. Relative
motion is L2 projected onto the cavity's displacement basis. Its transpose
maps generalized forces back to a distributed, work-conjugate beam reaction.
"""

from copy import deepcopy

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import lstsq, solve

from .foundation import motion_basis
from .mixed import dde, derivative, torch


def line_quadrature(knots, order=8, max_step=3.):
    """Gauss integration split at every forcing, region and motion-basis knot."""
    grid = np.unique(np.concatenate([
        np.linspace(a, b, int(np.ceil((b - a) / max_step)) + 1)
        for a, b in zip(knots[:-1], knots[1:])]))
    q, weights = np.polynomial.legendre.leggauss(order)
    half = np.diff(grid) / 2
    return ((grid[:-1, None] + grid[1:, None]) / 2 + half[:, None] * q).ravel(), (half[:, None] * weights).ravel()


class ProjectedFoundation:
    """E_ground = d^T C d/2-h^T d, d=P(w-g), P=M^-1 integral N^T(.) dx."""

    def __init__(self, stations, stiffness, x, weights):
        self.stations, self.stiffness = np.asarray(stations), np.asarray(stiffness)
        self.x, self.weights = np.asarray(x), np.asarray(weights)
        self.shape = motion_basis(self.x, self.stations)
        self.mass = self.shape.T @ (self.weights[:, None] * self.shape)
        self.project = solve(self.mass, self.shape.T * self.weights, assume_a='pos')
        self.force_map = solve(self.mass, self.stiffness, assume_a='pos')

    def coefficients(self, relative_motion):
        return self.project @ relative_motion

    def load_correction_from(self, reference, source_stiffness, source_correction):
        """Preserve ell=C_source P g+h_source when using this restraint.

        The reference is sampled at this foundation's quadrature points.
        Source and target matrices must use the same displacement basis.
        Columns may represent multiple excavation stages.
        """
        return (np.asarray(source_correction) +
                (np.asarray(source_stiffness) - self.stiffness) @ self.coefficients(reference))

    def pressure(self, x, coefficients, load_correction=None):
        """Soil energy derivative [kN/m]; the force on the beam is its negative."""
        force = self.force_map @ coefficients
        if load_correction is not None:
            force -= solve(self.mass, load_correction, assume_a='pos')
        return motion_basis(np.asarray(x), self.stations) @ force


class NonlocalPINN(torch.nn.Module):
    """w, phi, M, Q have shared region endpoints; M=Q=0 at free outer ends."""

    def __init__(self, edges, B, S, stations, stiffness, forcing_x, forcing_g,
                 width=32, layers=3, quadrature_order=8):
        super().__init__()
        self.edges, self.B, self.S = np.asarray(edges, dtype=float), B, S
        self.stations, self.stiffness = np.asarray(stations), np.asarray(stiffness)
        self.forcing_x, self.forcing_g = np.asarray(forcing_x), np.asarray(forcing_g)
        if not np.allclose(self.edges[[0, -1]], self.stations[[0, -1]]):
            raise ValueError('Beam ends must match the saved relative-motion support.')
        if self.forcing_x[0] > self.edges[0] or self.forcing_x[-1] < self.edges[-1]:
            raise ValueError('Free-field input does not cover the coupled beam.')
        knots = np.unique(np.r_[self.edges, self.stations, self.forcing_x])
        knots = knots[(knots >= self.edges[0]) & (knots <= self.edges[-1])]
        self.qx, self.qweights = line_quadrature(knots, quadrature_order)
        self.foundation = ProjectedFoundation(stations, stiffness, self.qx, self.qweights)
        effective_stiffness = np.trace(stiffness) / np.trace(self.foundation.mass)
        self.L = (B / effective_stiffness)**.25
        self.W = max(float(np.max(np.abs(forcing_g))), 1e-6)
        self.scales = np.array([self.W, self.W / self.L, B * self.W / self.L**2,
                                B * self.W / self.L**3])
        self.forcing = CubicSpline(forcing_x, forcing_g, extrapolate=False)
        self.load_correction = np.zeros(len(stations) - 2)
        self.nodes = torch.nn.Parameter(torch.zeros((len(edges), 4), dtype=torch.float64))
        mask = torch.ones_like(self.nodes)
        mask[[0, -1], 2:] = 0
        self.register_buffer('node_mask', mask)
        self.nets = torch.nn.ModuleList([
            dde.nn.FNN([1] + [width] * layers + [4], 'tanh', 'Glorot uniform')
            for _ in range(len(edges) - 1)])
        # Zero-bias tanh layers provide only odd hidden features. A nonzero
        # initial shift also supplies even terms for an output-only solve.
        # Transferred, trained hidden layers overwrite this initialization.
        for net in self.nets:
            for layer in net.linears[:-1]:
                torch.nn.init.uniform_(layer.bias, -.5, .5)

    def region(self, xhat, index):
        left, right = self.edges[index:index + 2] / self.L
        t = 2 * (xhat - left) / (right - left) - 1
        nodes = self.nodes * self.node_mask
        return ((1 - t) / 2 * nodes[index] + (1 + t) / 2 * nodes[index + 1] +
                (1 - t**2) * self.nets[index](t))

    def _linear_basis(self, x=None, derivatives=True):
        x = self.qx if x is None else np.asarray(x, dtype=float)
        if np.any((x < self.edges[0]) | (x > self.edges[-1])):
            raise ValueError('Basis coordinates must lie within the beam support.')
        active_nodes = np.argwhere(self.node_mask.cpu().numpy() != 0)
        node_columns = {tuple(key): j for j, key in enumerate(active_nodes)}
        feature_width = self.nets[0].linears[-1].in_features + 1
        n_columns = len(active_nodes) + len(self.nets) * 4 * feature_width
        values = np.zeros((len(x), 4, n_columns))
        gradients = np.zeros_like(values)
        regions = np.minimum(np.searchsorted(self.edges, x, side='right') - 1,
                             len(self.nets) - 1)
        for region, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            sel = regions == region
            if not np.any(sel):
                continue
            xt = torch.tensor((x[sel] / self.L)[:, None], requires_grad=derivatives)
            t = 2 * (xt - a / self.L) / ((b - a) / self.L) - 1
            hidden = t
            for layer in self.nets[region].linears[:-1]:
                hidden = torch.tanh(layer(hidden))
            bubble = (1 - t**2) * torch.cat([hidden, torch.ones_like(t)], dim=1)
            local = torch.cat([(1 - t) / 2, (1 + t) / 2, bubble], dim=1)
            v = local.detach().cpu().numpy()
            d = np.column_stack([
                torch.autograd.grad(local[:, j].sum(), xt, retain_graph=True)[0].detach().cpu().numpy().ravel()
                for j in range(local.shape[1])]) if derivatives else np.zeros_like(v)
            for state in range(4):
                for j in range(local.shape[1]):
                    if j < 2:
                        column = node_columns.get((region + j, state))
                        if column is None:
                            continue
                    else:
                        column = len(active_nodes) + region * 4 * feature_width + state * feature_width + j - 2
                    values[sel, state, column] = v[:, j]
                    gradients[sel, state, column] = d[:, j]
        return values, gradients

    def field_matrix(self, x):
        """Physical four-field sampling matrix for the trainable coefficients."""
        values, _ = self._linear_basis(x, derivatives=False)
        return values * self.scales[None, :, None]

    def linear_residual_system(self):
        """Weighted four-equation matrix and generalized-load right-hand side.

        The returned residual is A @ theta - F @ ell for zero reference lift.
        F acts on forces in kN, with the same adjoint projection as the ground.
        """
        values, matrix = self._linear_basis()
        matrix[:, 0] -= values[:, 1] + self.B / (self.S * self.L**2) * values[:, 3]
        matrix[:, 1] -= values[:, 2]
        matrix[:, 2] += values[:, 3]
        scale = self.L**4 / self.B
        projected = self.foundation.project @ values[:, 0]
        matrix[:, 3] -= scale * self.foundation.shape @ self.foundation.force_map @ projected
        source = np.zeros((len(self.qx), 4, len(self.stations) - 2))
        source[:, 3] = -scale / self.W * self.foundation.shape @ solve(
            self.foundation.mass, np.eye(len(self.stations) - 2), assume_a='pos')
        weights = np.sqrt(self.qweights / (4 * np.ptp(self.edges)))
        return ((matrix * weights[:, None, None]).reshape(-1, matrix.shape[-1]),
                (source * weights[:, None, None]).reshape(-1, source.shape[-1]))

    def polish_linear_outputs(self, forcing_fields=None, load_corrections=None):
        """Solve one or several loads on the same hidden features and scales.

        Multiple loads are columns sampled at forcing_x. One factorization
        serves all stages of a fixed linear soil/beam configuration. Optional
        load_corrections are generalized forces [kN], one column per load.
        """
        values, matrix = self._linear_basis()
        matrix[:, 0] -= values[:, 1] + self.B / (self.S * self.L**2) * values[:, 3]
        matrix[:, 1] -= values[:, 2]
        matrix[:, 2] += values[:, 3]
        scale = self.L**4 / self.B
        projected_w = self.foundation.project @ values[:, 0]
        matrix[:, 3] -= scale * self.foundation.shape @ self.foundation.force_map @ projected_w
        forcing_fields = self.forcing_g[:, None] if forcing_fields is None else np.asarray(forcing_fields)
        projected_g = self.foundation.coefficients(CubicSpline(self.forcing_x, forcing_fields)(self.qx) / self.W)
        source = np.zeros((len(self.qx), 4, forcing_fields.shape[1]))
        source[:, 3, :] = -scale * self.foundation.pressure(self.qx, projected_g)
        if load_corrections is not None:
            source[:, 3, :] += scale * self.foundation.pressure(
                self.qx, np.zeros_like(projected_g), np.asarray(load_corrections) / self.W)
        weights = np.sqrt(self.qweights / (4 * np.ptp(self.edges)))
        matrix = (matrix * weights[:, None, None]).reshape(-1, matrix.shape[-1])
        source = (source * weights[:, None, None]).reshape(-1, forcing_fields.shape[1])
        solution, _, rank, _ = lstsq(matrix, source, cond=1e-10, lapack_driver='gelsd')
        self.linear_solutions = solution
        correction = (np.zeros(len(self.stations) - 2) if load_corrections is None
                      else np.asarray(load_corrections)[:, 0])
        self.set_load_case(solution[:, 0], forcing_fields[:, 0], correction)
        return {'optimizer': 'linear_output_least_squares', 'columns': len(solution),
                'rank': int(rank), 'quadrature_points': len(self.qx),
                'svd_relative_cutoff': 1e-10,
                'loss_by_load': np.sum((matrix @ solution - source)**2, axis=0).tolist()}

    def set_linear_solution(self, solution):
        active_nodes = np.argwhere(self.node_mask.cpu().numpy() != 0)
        feature_width = self.nets[0].linears[-1].in_features + 1
        with torch.no_grad():
            for j, (node, state) in enumerate(active_nodes):
                self.nodes[node, state] = solution[j]
            for region, net in enumerate(self.nets):
                start = len(active_nodes) + region * 4 * feature_width
                coefficients = solution[start:start + 4 * feature_width].reshape(4, feature_width)
                net.linears[-1].weight.copy_(torch.tensor(coefficients[:, :-1]))
                net.linears[-1].bias.copy_(torch.tensor(coefficients[:, -1]))

    def set_load_case(self, solution, forcing_g, load_correction):
        """Activate a solved state together with both parts of its physical load."""
        self.set_linear_solution(solution)
        self.forcing_g = np.asarray(forcing_g).copy()
        self.forcing = CubicSpline(self.forcing_x, self.forcing_g, extrapolate=False)
        self.load_correction = np.asarray(load_correction).copy()

    def predict_fields(self, x):
        """Evaluate w, phi, M and Q at actual observation coordinates."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or np.any(~np.isfinite(x)):
            raise ValueError('Observation coordinates must be a finite one-dimensional array.')
        if np.any((x < self.edges[0]) | (x > self.edges[-1])):
            raise ValueError('Observation coordinates must lie within the beam support.')
        regions = np.searchsorted(self.edges, x, side='right') - 1
        regions = np.minimum(regions, len(self.nets) - 1)
        fields = np.empty((len(x), 4))
        with torch.no_grad():
            for i in np.unique(regions):
                selected = regions == i
                points = torch.tensor((x[selected] / self.L)[:, None])
                fields[selected] = self.region(points, int(i)).cpu().numpy() * self.scales
        return fields

    def with_forcing_gain(self, gain):
        """Return a copy with the current excavation load and response scaled.

        At fixed C, B and S, ell=C P g0+h scales only when both g0 and h
        scale. Scaling W and the four output scales propagates the gain to
        all beam fields, their derivatives and the work-conjugate reaction.
        The dimensionless neural coefficients and the prior stay unchanged.
        """
        gain = float(gain)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError('Forcing gain must be finite and positive.')
        updated = deepcopy(self)
        # The active stage spline can differ from the full-load input used
        # to choose W when several stages shared one output-layer solve.
        updated.forcing_g = gain * self.forcing(self.forcing_x)
        updated.forcing = CubicSpline(self.forcing_x, updated.forcing_g, extrapolate=False)
        updated.load_correction = gain * self.load_correction
        updated.W = gain * self.W
        updated.scales = gain * self.scales
        return updated

    def motion_coefficients(self):
        w = np.zeros_like(self.qx)
        for region, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            sel = (self.qx > a) & (self.qx < b)
            w[sel] = self.region(torch.tensor((self.qx[sel] / self.L)[:, None]), region)[:, 0].detach().numpy() * self.W
        return self.foundation.coefficients(w - self.forcing(self.qx))

    def evaluate(self, points_per_region=601):
        motion = self.motion_coefficients()
        rows = []
        for i, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            x = np.linspace(a, b, points_per_region)
            xt = torch.tensor((x / self.L)[:, None], requires_grad=True)
            y = self.region(xt, i)
            d = derivative(y, xt)
            fields = y.detach().numpy() * self.scales
            slope = d[:, 0].detach().numpy() * self.W / self.L
            bend = d[:, 1].detach().numpy() * self.W / self.L**2
            axis_curve = torch.autograd.grad(d[:, 0].sum(), xt)[0].detach().numpy().ravel() * self.W / self.L**2
            pressure = self.foundation.pressure(x, motion, self.load_correction)
            projected = motion_basis(x, self.stations) @ motion
            rows.append(np.column_stack([x, fields, slope, bend, axis_curve,
                                         self.forcing(x), projected, pressure, np.full_like(x, i)]))
        return np.concatenate(rows)

    def diagnostics(self):
        """Weak energy and rigid-translation balance at the integration points."""
        w = np.zeros_like(self.qx)
        energy = 0.
        residual_integral = np.zeros(4)
        motion = self.motion_coefficients()
        pressure = self.foundation.pressure(self.qx, motion, self.load_correction)
        for i, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            sel = (self.qx > a) & (self.qx < b)
            xt = torch.tensor((self.qx[sel] / self.L)[:, None], requires_grad=True)
            y = self.region(xt, i)
            d = derivative(y, xt).detach().numpy()
            y = y.detach().numpy()
            w[sel] = y[:, 0] * self.W
            phi = y[:, 1] * self.W / self.L
            slope = d[:, 0] * self.W / self.L
            bend = d[:, 1] * self.W / self.L**2
            energy += self.qweights[sel] @ (self.B * bend**2 + self.S * (slope - phi)**2)
            residual = d.copy()
            residual[:, 0] -= y[:, 1] + self.B / (self.S * self.L**2) * y[:, 3]
            residual[:, 1] -= y[:, 2]
            residual[:, 2] += y[:, 3]
            residual[:, 3] -= pressure[sel] * self.L**4 / (self.B * self.W)
            residual_integral += self.qweights[sel] @ residual**2
        pw = self.foundation.coefficients(w)
        pg = self.foundation.coefficients(self.forcing(self.qx))
        energy += pw @ self.stiffness @ pw
        work = pw @ (self.stiffness @ pg + self.load_correction)
        relative = w - self.forcing(self.qx)
        projected = self.foundation.shape @ motion
        central = abs(self.qx) <= 60
        return {'weak_energy_relative_gap': float((energy - work) / abs(work)) if work else float(energy),
                'integrated_tunnel_balance_kN': float(self.qweights @ pressure),
                'residual_rms_by_equation': np.sqrt(residual_integral / np.ptp(self.edges)).tolist(),
                'relative_motion_projection_rms_central_mm': float(1000 * np.sqrt(
                    np.average((relative[central] - projected[central])**2, weights=self.qweights[central]))),
                'motion_coefficients_m': motion.tolist()}
