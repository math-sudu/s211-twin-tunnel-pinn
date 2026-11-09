"""Wall-connected protection and spatially coupled longitudinal solvers."""

import numpy as np
from scipy.linalg import block_diag, solve
from scipy.sparse import csr_matrix, vstack

from .formation import FormedComponent
from .foundation import horizontal_surface_load, motion_basis
from .nonlocal_pinn import NonlocalPINN, line_quadrature


def axial_connection(basis, endpoints, young, area, reference=0.):
    """Axial member attached at two retained-wall points; units m, kN.

    Its extension is the relative displacement projected on its axis. Thus a
    horizontal member has no direct vertical attachment to either tunnel.
    """
    points = np.asarray(endpoints, dtype=float)
    direction = points[1] - points[0]
    length = np.linalg.norm(direction)
    if length <= 0 or young <= 0 or area <= 0:
        raise ValueError('Member length, modulus and area must be positive.')
    direction /= length
    probes = basis.probes(points.T).tocsr()
    H = sum(direction[j] * (probes[2*j+1] - probes[2*j]) for j in range(3))
    return FormedComponent(H, [[young * area / length]], [reference])


def member_weight(basis, endpoints, area, unit_weight):
    """Consistent two-end vertical loads for a horizontal member's self weight."""
    points = np.asarray(endpoints, dtype=float)
    probes = basis.probes(points.T).tocsr()
    weight = np.linalg.norm(points[1] - points[0]) * area * unit_weight
    return np.asarray(-(probes[4] + probes[5]).toarray()).ravel() * weight / 2


def connection_operators(basis, members):
    """Build all wall attachments in one point-location operation."""
    points = np.asarray([p for member in members for p in member['endpoints_m']])
    probes = basis.probes(points.T).tocsr()
    H, weights, stiffness = [], [], []
    n = len(points)
    for i, member in enumerate(members):
        vector = points[2*i+1]-points[2*i]
        length = np.linalg.norm(vector)
        direction = vector/length
        H.append(sum(direction[j]*(probes[j*n+2*i+1]-probes[j*n+2*i]) for j in range(3)))
        stiffness.append(member['young_kPa']*member['area_m2']/length)
        weights.append(-(probes[2*n+2*i]+probes[2*n+2*i+1]) *
                       (member['unit_weight_kN_m3']*member['area_m2']*length/2))
    return vstack(H).tocsr(), np.asarray(stiffness), vstack(weights).tocsr()


def casting_loads(basis, config, walls, loads):
    """Documented pour resultants on retained wall rows or exterior ground."""
    thickness = walls[0][1]-walls[0][0]
    forces = []
    for group in loads['casting_groups']:
        c, d = np.asarray(group['road_chainage_m'])-loads['road_origin_chainage_m']
        rectangles = [[x-thickness/2,x+thickness/2,c,d] for x in config['x_divisions_m']]
        force = horizontal_surface_load(basis,rectangles,np.ones(len(rectangles)),
                                        config['ground']['surface_height_m'])
        force *= -group['volume_m3']*loads['concrete_unit_weight_kN_m3']/force.sum()
        forces.append(force)
    return np.column_stack(forces)


def reduce_ovalization(stations, operator, load, ring_bending, radius):
    """Eliminate two inextensional cos(2 theta) ring modes after soil reduction.

    Ring energy per length is 9*pi*B*q^2/(2*R^3). Return translation
    stiffness/load and q=q0+T*w. A zero B case tests the soil-only restraint;
    an intact-ring B supplies the other representative section condition.
    """
    n = operator.shape[0]//2
    mass, _ = line_matrices(stations)
    ring = 9*np.pi*ring_bending/radius**3*block_diag(mass,mass)
    cross = operator[:n,n:]
    q0 = solve(operator[n:,n:]+ring,load[n:],assume_a='pos')
    T = -solve(operator[n:,n:]+ring,operator[n:,:n],assume_a='pos')
    return operator[:n,:n]+cross@T, load[:n]-cross@q0, q0, T


def line_matrices(stations):
    n = len(stations)
    mass, gradient = np.zeros((n, n)), np.zeros((n, n))
    for i, h in enumerate(np.diff(stations)):
        mass[i:i+2, i:i+2] += h / 6 * np.array([[2., 1.], [1., 2.]])
        gradient[i:i+2, i:i+2] += np.array([[1., -1.], [-1., 1.]]) / h
    return mass[1:-1, 1:-1], gradient[1:-1, 1:-1]


class BeamReference:
    """Independent linear Timoshenko finite elements with reduced shear integration.

    Free end moments/shears are natural conditions. The same L2 displacement
    projection couples both beams to the soil; no lining energy enters the soil.
    """
    def __init__(self, stations, B, S, operator, max_step=1.5):
        self.grid = np.unique(np.concatenate([np.linspace(a, b, int(np.ceil((b-a)/max_step))+1)
                                              for a, b in zip(stations[:-1], stations[1:])]))
        self.B, self.S, self.stations = B, S, np.asarray(stations)
        grid, n = self.grid, len(self.grid)
        K = np.zeros((2*n, 2*n))
        load_project = np.zeros((len(stations)-2, 2*n))
        q, qw = np.polynomial.legendre.leggauss(3)
        for i, h in enumerate(np.diff(grid)):
            ids = np.array([2*i, 2*i+1, 2*i+2, 2*i+3])
            bend, shear = np.array([0., -1/h, 0., 1/h]), np.array([-1/h, -.5, 1/h, -.5])
            K[np.ix_(ids, ids)] += h*(B*np.outer(bend, bend) + S*np.outer(shear, shear))
            x = (grid[i]+grid[i+1])/2+h*q/2
            shape = np.column_stack([(1-q)/2, (1+q)/2])
            load_project[:, ids[[0, 2]]] += motion_basis(x, stations).T @ ((h*qw/2)[:, None]*shape)
        mass, _ = line_matrices(stations)
        self.P = solve(mass, load_project, assume_a='pos')
        self.project = block_diag(self.P, self.P)
        self.stiffness = block_diag(K, K) + self.project.T @ operator @ self.project

    def solve(self, load):
        self.solution = solve(self.stiffness, self.project.T @ load, assume_a='pos')
        self.coefficients = self.project @ self.solution
        return self.coefficients

    def fields(self, x, line):
        grid, x = self.grid, np.asarray(x)
        values = self.solution.reshape(2, len(grid), 2)[line]
        i = np.searchsorted(grid, x, side='right').clip(1, len(grid)-1)-1
        h = grid[i+1]-grid[i]
        t = (x-grid[i])/h
        v = (1-t[:, None])*values[i]+t[:, None]*values[i+1]
        deriv = (values[i+1]-values[i])/h[:, None]
        M = self.B*deriv[:, 1]
        Q = self.S*(deriv[:, 0]-(values[i, 1]+values[i+1, 1])/2)
        return np.column_stack([v, M, Q])


class CoupledPINN:
    """Two existing four-field networks joined by their soil reaction blocks.

    Output-only solves provide linear response maps. A small coupled solve
    then enforces interaction exactly within those neural trial spaces.
    """
    def __init__(self, stations, edges, B, S, operator, checkpoint=None,
                 original_edges=None, width=32, layers=3):
        self.operator, self.models, self.responses, self.records = operator, [], [], []
        n = len(stations)-2
        self.n = n
        for line in range(2):
            C = operator[line*n:(line+1)*n, line*n:(line+1)*n]
            model = NonlocalPINN(edges, B, S, stations, C, np.asarray(stations),
                                np.zeros(len(stations)), width=width, layers=layers)
            if checkpoint is not None:
                # Only hidden features are transferred; region endpoints,
                # scales and output coefficients belong to the current model.
                for j, midpoint in enumerate((np.asarray(edges[:-1])+edges[1:])/2):
                    old = np.searchsorted(original_edges, midpoint)-1
                    for layer in range(layers):
                        model.nets[j].linears[layer].weight.data.copy_(checkpoint[f'nets.{old}.linears.{layer}.weight'])
                        model.nets[j].linears[layer].bias.data.copy_(checkpoint[f'nets.{old}.linears.{layer}.bias'])
            record = model.polish_linear_outputs(np.zeros((len(stations), n)), np.eye(n))
            values, _ = model._linear_basis()
            response = model.foundation.project @ (values[:, 0] @ model.linear_solutions * model.W)
            self.models.append(model)
            self.responses.append(response)
            self.records.append(record)
        R1, R2 = self.responses
        self.coupling = np.block([[np.eye(n), R1 @ operator[:n, n:]],
                                  [R2 @ operator[n:, :n], np.eye(n)]])
        self.motion_response = solve(self.coupling, block_diag(R1, R2))
        self.output_response = []
        for line, model in enumerate(self.models):
            own = slice(line*n, (line+1)*n)
            other = slice((1-line)*n, (2-line)*n)
            effective = np.eye(2*n)[own] - operator[own, other] @ self.motion_response[other]
            self.output_response.append(model.linear_solutions @ effective)

    def solve(self, load):
        n = self.n
        coefficients = solve(self.coupling, np.r_[self.responses[0] @ load[:n],
                                                  self.responses[1] @ load[n:]])
        for line, model in enumerate(self.models):
            own, other = slice(line*n, (line+1)*n), slice((1-line)*n, (2-line)*n)
            effective = load[own]-self.operator[own, other] @ coefficients[other]
            model.set_load_case(model.linear_solutions @ effective, np.zeros(len(model.forcing_x)), effective)
        self.coefficients = coefficients
        return coefficients

    def fields(self, x, line):
        return self.models[line].predict_fields(x)

    def field_response(self, x):
        """Both neural fields per unit generalized force, including cross-line work."""
        return np.stack([model.field_matrix(x) @ response
                         for model, response in zip(self.models, self.output_response)])

    def residual_response(self, displacement_scale=1e-3):
        """Full mixed residual on the coupled neural response space.

        A @ q - F @ ell retains all four equations and both off-diagonal
        ground blocks. Observations can change q without changing the supplied
        construction load ell; the resulting physics mismatch is penalized.
        """
        matrices, sources = [], []
        for line, model in enumerate(self.models):
            own = slice(line*self.n, (line+1)*self.n)
            other = slice((1-line)*self.n, (2-line)*self.n)
            local, force = model.linear_residual_system()
            matrix = local @ self.output_response[line]
            matrix += force @ self.operator[own, other] @ self.motion_response[other]
            source = np.zeros((len(force), 2*self.n))
            source[:, own] = force
            # Each line has the same physical displacement normalization;
            # the residual's length scale remains that of its ground block.
            matrices.append(matrix * model.W / displacement_scale / np.sqrt(2))
            sources.append(source * model.W / displacement_scale / np.sqrt(2))
        return np.vstack(matrices), np.vstack(sources)
