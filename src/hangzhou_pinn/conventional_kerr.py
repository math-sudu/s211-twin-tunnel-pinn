"""Conventional Kerr beam with independently integrated half-space stresses.

The foundation follows the simplified elastic-continuum coefficients in
Huang et al. (2012), section 2.1, and Feng et al. (2022), equation (20).
The vertical Mindlin kernel follows Feng et al., equation (22). Coordinates
use positive depth below the original surface; pressure and motion are
positive upward. No continuum stiffness or displacement is an input.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def kerr_parameters(young, poisson, burial):
    """Return c, k [kN/m^3] and G [kN/m] from E [kPa] and burial [m]."""
    lower = 4 * young / (3 * burial)
    return 3 * lower, lower, 2 * young * burial / (9 * (1 + poisson))


def mindlin_vertical_stress(dx, dy, depth, source_depth, poisson):
    """Vertical stress per vertical unit force, with the same force sign.

    At source_depth=0 this reduces to 3*z^3/(2*pi*R^5), the Boussinesq
    stress kernel. The source is a vertical point force in a half-space.
    """
    z, d = np.asarray(depth), np.asarray(source_depth)
    minus, plus = z - d, z + d
    r1 = np.sqrt(np.asarray(dx)**2 + np.asarray(dy)**2 + minus**2)
    r2 = np.sqrt(np.asarray(dx)**2 + np.asarray(dy)**2 + plus**2)
    if np.any(r1 == 0):
        raise ValueError('Mindlin observation coincides with a point load.')
    return ((1 - 2 * poisson) * minus * (r1**-3 - r2**-3)
            + 3 * minus**3 / r1**5
            + (3 * (3 - 4 * poisson) * z * plus**2
               - 3 * d * plus * (5 * z - d)) / r2**5
            + 30 * d * z * plus**3 / r2**7) / (8 * np.pi * (1 - poisson))


def rectangular_stress(x, y, depth, rectangle, source_depth, pressure,
                       poisson, order=16):
    """Integrate uniform vertical pressure over a horizontal rectangle."""
    a, b, c, d = rectangle
    nodes, weights = leggauss(order)
    sx, sy = np.meshgrid((a + b) / 2 + nodes * (b - a) / 2,
                         (c + d) / 2 + nodes * (d - c) / 2)
    force = (np.outer(weights, weights) * (b-a) * (d-c) * pressure / 4).ravel()
    kernel = mindlin_vertical_stress(np.asarray(x)[:, None] - sx.ravel(),
                                    np.asarray(y)[:, None] - sy.ravel(),
                                    depth, source_depth, poisson)
    return kernel @ force


class KerrBeam:
    """Linear Timoshenko beam and two spring layers on a common line mesh.

    Unknowns at each node are (w, phi, v). Beam shear uses reduced
    integration; spring energies and applied loads use consistent line
    mass matrices. Free M, Q and shear-layer flux are natural end conditions.
    """

    def __init__(self, grid, bending, shear, diameter, young, poisson, burial):
        self.grid = np.asarray(grid, dtype=float)
        self.bending, self.shear = bending, shear
        self.parameters = kerr_parameters(young, poisson, burial)
        upper, lower, layer = self.parameters
        n = len(self.grid)
        rows, columns, entries = [], [], []
        mass_rows, mass_columns, mass_entries = [], [], []
        for i, h in enumerate(np.diff(self.grid)):
            mass = h / 6 * np.array([[2., 1.], [1., 2.]])
            gradient = np.array([[1., -1.], [-1., 1.]]) / h
            curvature = np.array([0., -1/h, 0., 0., 1/h, 0.])
            shear_strain = np.array([-1/h, -.5, 0., 1/h, -.5, 0.])
            local = h * (bending * np.outer(curvature, curvature)
                         + shear * np.outer(shear_strain, shear_strain))
            w, v = [0, 3], [2, 5]
            local[np.ix_(w, w)] += diameter * upper * mass
            local[np.ix_(w, v)] -= diameter * upper * mass
            local[np.ix_(v, w)] -= diameter * upper * mass
            local[np.ix_(v, v)] += diameter * ((upper+lower) * mass + layer * gradient)
            ids = np.arange(3*i, 3*i+6)
            rows.extend(np.repeat(ids, 6)); columns.extend(np.tile(ids, 6))
            entries.extend(local.ravel())
            mids = np.array([i, i+1])
            mass_rows.extend(np.repeat(mids, 2)); mass_columns.extend(np.tile(mids, 2))
            mass_entries.extend((diameter * mass).ravel())
        stiffness = coo_matrix((entries, (rows, columns)), shape=(3*n, 3*n)).tocsc()
        self.load_mass = coo_matrix((mass_entries, (mass_rows, mass_columns)), shape=(n, n)).tocsr()
        self.factor = splu(stiffness)

    def solve(self, pressure):
        """Solve for nodal pressure [kPa]; return nodal (w, phi, v)."""
        rhs = np.zeros(3 * len(self.grid))
        rhs[::3] = self.load_mass @ np.asarray(pressure)
        self.solution = self.factor.solve(rhs).reshape(-1, 3)
        return self.solution

    def fields(self, x):
        """Return (w, phi, M, Q) using the manuscript's sign convention."""
        x = np.asarray(x)
        i = np.searchsorted(self.grid, x, side='right').clip(1, len(self.grid)-1) - 1
        h = self.grid[i+1] - self.grid[i]
        fraction = (x - self.grid[i]) / h
        values = ((1-fraction[:, None]) * self.solution[i]
                  + fraction[:, None] * self.solution[i+1])
        derivative = (self.solution[i+1] - self.solution[i]) / h[:, None]
        moment = self.bending * derivative[:, 1]
        force = self.shear * (derivative[:, 0]
                              - (self.solution[i, 1] + self.solution[i+1, 1])/2)
        return np.column_stack([values[:, :2], moment, force])
