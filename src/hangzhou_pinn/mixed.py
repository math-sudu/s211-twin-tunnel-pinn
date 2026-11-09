"""Six-state PINN with exact displacement/flux continuity at region boundaries.

The retained DeepXDE beam example supplies FNN construction and AD training
conventions.  The energy, mixed residual, scales and interface representation
here are the project's own adaptation.
"""

import os
os.environ.setdefault('DDE_BACKEND', 'pytorch')
import deepxde as dde
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import CubicSpline
import torch

dde.config.set_default_float('float64')
torch.set_default_device('cpu')
torch.set_num_threads(1)


def derivative(y, x):
    return torch.cat([torch.autograd.grad(y[:, j].sum(), x, create_graph=True,
                                         retain_graph=True)[0] for j in range(y.shape[1])], dim=1)


class MixedPINN(torch.nn.Module):
    """Region bubbles share all six endpoint states; outer M,Q,H are zero."""

    def __init__(self, edges, coefficients, forcing_x, forcing_g, width=32, layers=3,
                 forcing_interpolation='linear'):
        super().__init__()
        self.edges = np.asarray(edges, dtype=float)
        self.coefficients = coefficients
        self.forcing_interpolation = forcing_interpolation
        c = coefficients[0]
        keq = c['ku'] * c['kl'] / (c['ku'] + c['kl'])
        self.L = (c['B'] / keq)**0.25
        self.W = max(float(np.max(np.abs(forcing_g))), 1e-6)
        self.B, self.T = c['B'], c['T']
        self.scales = np.array([self.W, self.W / self.L, self.B * self.W / self.L**2,
                                self.B * self.W / self.L**3, self.W, self.T * self.W / self.L])
        self.register_buffer('forcing_x', torch.tensor(forcing_x, dtype=torch.float64))
        self.register_buffer('forcing_g', torch.tensor(forcing_g / self.W, dtype=torch.float64))
        self.register_buffer('forcing_polynomial', torch.tensor(CubicSpline(forcing_x, forcing_g / self.W).c), persistent=False)
        self.nodes = torch.nn.Parameter(torch.zeros((len(edges), 6), dtype=torch.float64))
        mask = torch.ones_like(self.nodes)
        mask[0, [2, 3, 5]] = 0
        mask[-1, [2, 3, 5]] = 0
        self.register_buffer('node_mask', mask)
        self.nets = torch.nn.ModuleList([
            dde.nn.FNN([1] + [width] * layers + [6], 'tanh', 'Glorot uniform')
            for _ in coefficients])

    def region(self, xhat, index):
        left, right = self.edges[index:index + 2] / self.L
        t = 2 * (xhat - left) / (right - left) - 1
        nodes = self.nodes * self.node_mask
        return ((1 - t) / 2 * nodes[index] + (1 + t) / 2 * nodes[index + 1] +
                (1 - t**2) * self.nets[index](t))

    def forcing(self, xhat):
        x = (xhat.detach() * self.L).reshape(-1)
        if x.min() < self.forcing_x[0] - 1e-9 or x.max() > self.forcing_x[-1] + 1e-9:
            raise ValueError('Free-field input does not cover the PINN domain.')
        k = torch.searchsorted(self.forcing_x, x.contiguous()).clamp(1, len(self.forcing_x) - 1)
        offset = x - self.forcing_x[k - 1]
        if self.forcing_interpolation == 'linear':
            t = offset / (self.forcing_x[k] - self.forcing_x[k - 1])
            return (self.forcing_g[k - 1] + t * (self.forcing_g[k] - self.forcing_g[k - 1]))[:, None]
        c = self.forcing_polynomial[:, k - 1]
        return (((c[0] * offset + c[1]) * offset + c[2]) * offset + c[3])[:, None]

    def coupling(self, index):
        c = self.coefficients[index]
        upper = c['ku'] * self.L**4 / self.B
        lower = c['kl'] * self.L**2 / self.T
        transfer = c['ku'] * self.L**2 / self.T
        return np.array([
            [0, -1, 0, -self.B / (c['S'] * self.L**2), 0, 0],
            [0, 0, -self.B / c['B'], 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [-upper, 0, 0, 0, upper, 0],
            [0, 0, 0, 0, 0, -self.T / c['T']],
            [transfer, 0, 0, 0, -(lower + transfer), 0],
        ])

    def residual(self, xhat, y, index, ghat=None, qdhat=0):
        """qdhat = qd*L^4/(B*W); spatial coefficients are never differentiated."""
        g = self.forcing(xhat) if ghat is None else ghat
        matrix = torch.tensor(self.coupling(index))
        source = torch.cat([g * 0, g * 0, g * 0, -matrix[3, 0] * g + qdhat,
                            g * 0, -matrix[5, 0] * g], dim=1)
        return derivative(y, xhat) + y @ matrix.T + source

    def polish_linear_outputs(self, points_per_region=256):
        """Solve the linear output/node coefficients on the learned features, qd=0.

        This uses the same PINN basis and residual. The frozen hidden layers
        preserve nonlinear feature learning; least squares removes incomplete
        output-layer optimization for this linear mechanical operator.
        """
        active_nodes = np.argwhere(self.node_mask.cpu().numpy() != 0)
        node_columns = {tuple(key): j for j, key in enumerate(active_nodes)}
        n_nodes = len(active_nodes)
        feature_width = self.nets[0].linears[-1].in_features + 1
        n_columns = n_nodes + len(self.nets) * 6 * feature_width
        matrices, sources = [], []
        for region, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            x = (a + (np.arange(points_per_region) + 0.5) * (b - a) / points_per_region) / self.L
            xt = torch.tensor(x[:, None], requires_grad=True)
            t = 2 * (xt - a / self.L) / ((b - a) / self.L) - 1
            hidden = t
            for layer in self.nets[region].linears[:-1]:
                hidden = torch.tanh(layer(hidden))
            bubble = (1 - t**2) * torch.cat([hidden, torch.ones_like(t)], dim=1)
            local_basis = torch.cat([(1 - t) / 2, (1 + t) / 2, bubble], dim=1)
            values = local_basis.detach().cpu().numpy()
            gradients = np.column_stack([
                torch.autograd.grad(local_basis[:, j].sum(), xt, retain_graph=True)[0].detach().cpu().numpy().ravel()
                for j in range(local_basis.shape[1])])
            matrix = np.zeros((points_per_region, 6, n_columns))
            coupling = self.coupling(region)
            for state in range(6):
                for j in range(local_basis.shape[1]):
                    if j < 2:
                        column = node_columns.get((region + j, state))
                        if column is None:
                            continue
                    else:
                        column = n_nodes + region * 6 * feature_width + state * feature_width + j - 2
                    matrix[:, :, column] += values[:, j, None] * coupling[None, :, state]
                    matrix[:, state, column] += gradients[:, j]
            forcing = self.forcing(xt).detach().cpu().numpy().ravel()
            source = np.zeros((points_per_region, 6))
            source[:, 3], source[:, 5] = coupling[3, 0] * forcing, coupling[5, 0] * forcing
            weight = np.sqrt((b - a) / np.ptp(self.edges) / (6 * points_per_region))
            matrices.append(matrix.reshape(-1, n_columns) * weight)
            sources.append(source.ravel() * weight)
        matrix, source = np.vstack(matrices), np.concatenate(sources)
        # Discard numerically dependent feature columns before their huge
        # coefficients amplify roundoff in spatial derivatives.
        solution, _, rank, singular = lstsq(matrix, source, cond=1e-10, lapack_driver='gelsd')
        with torch.no_grad():
            for j, (node, state) in enumerate(active_nodes):
                self.nodes[node, state] = solution[j]
            for region, net in enumerate(self.nets):
                start = n_nodes + region * 6 * feature_width
                coefficients = solution[start:start + 6 * feature_width].reshape(6, feature_width)
                net.linears[-1].weight.copy_(torch.tensor(coefficients[:, :-1]))
                net.linears[-1].bias.copy_(torch.tensor(coefficients[:, -1]))
        return {'optimizer': 'linear_output_least_squares', 'points_per_region': points_per_region,
                'columns': n_columns, 'rank': int(rank), 'svd_relative_cutoff': 1e-10,
                'loss': float(np.sum((matrix @ solution - source)**2)),
                'smallest_singular_value': float(singular[-1])}

    def train_physics(self, config):
        n = config['points_per_region']
        points = []
        for a, b in zip(self.edges[:-1], self.edges[1:]):
            x = (a + (np.arange(n) + 0.5) * (b - a) / n) / self.L
            points.append(torch.tensor(x[:, None], requires_grad=True))
        lengths = np.diff(self.edges) / np.ptp(self.edges)

        def loss():
            total = 0
            for i, x in enumerate(points):
                R = self.residual(x, self.region(x, i), i)
                total = total + lengths[i] * (R**2).mean()
            return total

        history = []
        adam = torch.optim.Adam(self.parameters(), lr=config['adam_lr'])
        for step in range(config['adam_steps']):
            adam.zero_grad(); value = loss(); value.backward(); adam.step()
            if step % 250 == 0:
                history.append({'optimizer': 'adam', 'step': step, 'loss': value.item()})
                print(f'Adam {step}: loss={value.item():.6g}', flush=True)
        optimizer = torch.optim.LBFGS(self.parameters(), max_iter=config['lbfgs_steps'],
                                     line_search_fn='strong_wolfe', tolerance_grad=1e-9,
                                     tolerance_change=1e-12)
        evaluations = 0

        def closure():
            nonlocal evaluations
            optimizer.zero_grad(); value = loss(); value.backward(); evaluations += 1
            if evaluations % 250 == 0:
                print(f'L-BFGS evaluation {evaluations}: loss={value.item():.6g}', flush=True)
            return value

        optimizer.step(closure)
        history.append({'optimizer': 'lbfgs', 'evaluations': evaluations, 'loss': loss().item()})
        return history

    def evaluate(self, points_per_region=301):
        rows, residuals = [], []
        for i, (a, b) in enumerate(zip(self.edges[:-1], self.edges[1:])):
            x = np.linspace(a, b, points_per_region)
            xt = torch.tensor((x / self.L)[:, None], requires_grad=True)
            y = self.region(xt, i)
            d = derivative(y, xt)
            r = self.residual(xt, y, i).detach().cpu().numpy()
            fields = y.detach().cpu().numpy() * self.scales
            slope = d[:, 0].detach().cpu().numpy() * self.W / self.L
            bending_curvature = d[:, 1].detach().cpu().numpy() * self.W / self.L**2
            axis_curvature = torch.autograd.grad(d[:, 0].sum(), xt)[0].detach().cpu().numpy().ravel() * self.W / self.L**2
            rows.append(np.column_stack([x, fields, slope, bending_curvature, axis_curvature,
                                         np.full_like(x, i)]))
            residuals.append(r)
        return np.concatenate(rows), np.concatenate(residuals)

    def diagnostics(self, profile, residual):
        x, w, phi, M, Q, r, H, slope, bend, axis_curve, region = profile.T
        energy = work = balance_q = balance_h = 0.0
        for i, c in enumerate(self.coefficients):
            sel = region == i
            xi = x[sel]
            g = self.forcing(torch.tensor((xi / self.L)[:, None])).cpu().numpy().ravel() * self.W
            rel = w[sel] - g - r[sel]
            # Weak energy test uses the free-end solution as its test function.
            density = c['B'] * bend[sel]**2 + c['S'] * (slope[sel] - phi[sel])**2
            density += c['ku'] * (w[sel] - r[sel])**2 + c['kl'] * r[sel]**2 + H[sel]**2 / c['T']
            energy += np.trapezoid(density, xi)
            work += np.trapezoid(c['ku'] * g * (w[sel] - r[sel]), xi)
            balance_q += np.trapezoid(c['ku'] * rel, xi)
            balance_h += np.trapezoid(c['kl'] * r[sel] - c['ku'] * rel, xi)
        j = int(np.argmax(w))
        return {'peak_uplift_mm': float(w[j] * 1000), 'peak_x_m': float(x[j]),
                'peak_abs_moment_kN_m': float(np.max(np.abs(M))),
                'peak_abs_axis_curvature_per_m': float(np.max(np.abs(axis_curve))),
                'weak_energy_relative_gap': float((energy - work) / max(abs(work), 1e-30)),
                'integrated_tunnel_balance_kN': float(balance_q),
                'integrated_layer_balance_kN': float(balance_h),
                'residual_rms_by_equation': np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                'end_fluxes': profile[[0, -1]][:, [3, 4, 6]].tolist()}
