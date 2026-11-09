"""Jointly train S211 neural states on construction physics and monitoring."""

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy.linalg import block_diag, solve
from hangzhou_pinn.conventional_kerr import KerrBeam
from hangzhou_pinn.foundation import motion_basis
from hangzhou_pinn.joint import JointObjective, history_system, select_weight
from hangzhou_pinn.mixed import torch, dde
from hangzhou_pinn.protection import CoupledPINN, line_matrices
from run_conventional_kerr import source_loading
from run_protection_stages import write_csv, write_json

STAGES = ('july', 'cut_ab', 'august')


def read_npz(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def observations(settings, physical):
    with (ROOT / settings['monitoring']).open(encoding='utf-8-sig', newline='') as handle:
        source = list(csv.DictReader(handle))
    by_id = {row['record_id']: row for row in source}
    rows = []
    for row in source:
        month = row['report_date'][:7]
        if (month not in settings['months'] or row['interval_usable'] != 'True' or
                row['observable'] != 'track_bed_vertical_movement'):
            continue
        x = float(row['chainage_m']) - physical['pit_origin_chainage_m']
        group = int(np.floor((x-settings['spatial_group_origin_m']) / settings['spatial_group_width_m']))
        rows.append({'record_id': row['record_id'], 'previous_record_id': row['previous_record_id'],
                     'baseline_date': by_id[row['previous_record_id']]['observation_date'],
                     'target_date': row['observation_date'], 'month': month,
                     'sensor': row['sensor'], 'line': row['line'], 'point_id': row['point_id'],
                     'x_m': x, 'observed_mm': float(row['analysis_change_mm']),
                     'spatial_group': group, 'validation_fold': group % settings['validation_folds'],
                     'split': 'test' if group in settings['test_groups'] else 'development'})
    return rows


def beam_stiffness(physical):
    b = physical['beam']
    area = np.pi * (b['outer_diameter_m']**2-b['inner_diameter_m']**2) / 4
    inertia = np.pi * (b['outer_diameter_m']**4-b['inner_diameter_m']**4) / 64
    return (b['concrete_modulus_kPa'] * inertia * b['bending_reduction'],
            b['concrete_modulus_kPa'] / (2*(1+b['poisson_ratio'])) * area *
            b['shear_reduction'] * b['shear_correction'])


def neural_problem(settings, physical, rows):
    source, water = ROOT / settings['physical_results'], ROOT / settings['water_results']
    geometry = json.loads((source / 'model_geometry.json').read_bytes())
    stations = np.asarray(geometry['stations_m'])
    B, S = beam_stiffness(physical)
    checkpoint = torch.load(ROOT / physical['feature_checkpoint'], weights_only=True)
    dde.config.set_random_seed(20261030)
    positions = np.array([row['x_m'] for row in rows])
    sampled = np.linspace(stations[0], stations[-1], 601)
    initial = read_npz(source / 'together_july_continuum_pinn.npz')['reference_extensions_m']
    initial += read_npz(water / 'july.npz')['water_reference_extensions_m']
    stages, solvers, response, profiles = [], [], [], []
    for name in STAGES:
        started = perf_counter()
        data = read_npz(source / f'{name}_condensed.npz')
        added = read_npz(water / f'{name}_water_load.npz')
        data['load0'] = data['load0'] + added['load']
        data['connection_particular'] = data['connection_particular'] + added['connection_particular']
        solver = CoupledPINN(stations, physical['pinn_edges_m'], B, S, data['operator'],
                             checkpoint, stations)
        A, F = solver.residual_response(settings['displacement_scale_m'])
        data.update(residual_matrix=A, force_matrix=F, motion_response=solver.motion_response)
        stages.append(data)
        solvers.append(solver)
        response.append(solver.field_response(positions))
        profiles.append(solver.field_response(sampled))
        print(f'Prepared {name} coupled neural response in {perf_counter()-started:.1f} s', flush=True)
    A, target, history = history_system(stages, initial)
    size = 2 * (len(stations)-2)
    prior = np.zeros(size * len(stages))
    for j, state in enumerate(history):
        prior[j*size:(j+1)*size] = state['load_offset'] + state['load_map'] @ prior
    gravity = read_npz(ROOT / settings['july_gravity'])['load']
    june_load = prior[:size] - gravity.sum(axis=1)
    june = response[0] @ june_load
    manual_extra = response[0] @ (gravity[:, 2] * 85 / 212)
    obs = np.zeros((len(rows), len(prior)))
    offset = np.zeros(len(rows))
    for i, row in enumerate(rows):
        line = 0 if row['line'] == 'up' else 1
        if row['month'] == '2026-07':
            obs[i, :size] = response[0][line, i, 0] * 1000
            offset[i] = -june[line, i, 0] * 1000
            if row['sensor'] == 'manual':
                offset[i] += manual_extra[line, i, 0] * 1000
        else:
            obs[i, 2*size:3*size] = response[2][line, i, 0] * 1000
            obs[i, :size] = -response[0][line, i, 0] * 1000
    payload = dict(stations=stations, solvers=solvers, stages=stages, history=history,
                   profiles=profiles, sampled=sampled, size=size, prior=prior)
    return A, target, obs, offset, payload


def kerr_problem(settings, physical, rows, neural):
    """Same observations and force-residual norm, independent Kerr mechanics."""
    config = copy.deepcopy(physical)
    config['ground']['effective_unit_weight_kN_m3'] += 9.81
    casting = json.loads((ROOT / config['casting_loading_ref']).read_bytes())
    grid = np.arange(-150., 150.25, .5)
    gravity, extra, pressures, _, _ = source_loading(config, casting, grid, 16)
    parameters = json.loads((ROOT / settings['physical_results'] /
                             'conventional_kerr/parameters.json').read_bytes())
    B, S = beam_stiffness(config)
    diameter = config['beam']['outer_diameter_m']
    kerr = KerrBeam(grid, B, S, diameter, parameters['effective_modulus_MPa']*1000,
                    config['ground']['poisson_ratio'], 17.8)
    stations, size = neural['stations'], neural['size']
    mass, _ = line_matrices(stations)
    force_to_pressure = motion_basis(grid, stations) @ solve(mass, np.eye(size//2), assume_a='pos') / diameter
    positions = np.array([row['x_m'] for row in rows])
    field_map = np.empty((len(rows), 4, size//2))
    for j in range(size//2):
        kerr.solve(force_to_pressure[:, j])
        field_map[:, :, j] = kerr.fields(positions)
    prior_fields = {}
    for name, pressure in {**{s: pressures[s] for s in STAGES},
                           'july_auto': gravity, 'july_manual': gravity+extra}.items():
        fields = []
        for line in range(2):
            kerr.solve(pressure[line])
            fields.append(kerr.fields(positions))
        prior_fields[name] = np.stack(fields)
    obs = np.zeros((len(rows), 3*size))
    offset = np.zeros(len(rows))
    for i, row in enumerate(rows):
        line = 0 if row['line'] == 'up' else 1
        own = slice(line*size//2, (line+1)*size//2)
        if row['month'] == '2026-07':
            obs[i, own] = field_map[i, 0]*1000
            key = 'july_auto' if row['sensor'] == 'automated' else 'july_manual'
            offset[i] = prior_fields[key][line, i, 0]*1000
        else:
            obs[i, own] = -field_map[i, 0]*1000
            obs[i, slice(2*size+own.start, 2*size+own.stop)] = field_map[i, 0]*1000
            offset[i] = (prior_fields['august'][line, i, 0]-prior_fields['july'][line, i, 0])*1000
    # Same distributed-force norm and dimensional normalization as the neural
    # comparison; no continuum displacement or loading enters the Kerr response.
    A = block_diag(*[data['force_matrix'] for data in neural['stages']]) / np.sqrt(3)
    return A, np.zeros(len(A)), obs, offset


def metrics(rows, predictions, prior, mask):
    results = []
    groups = sorted({(r['month'], r['sensor'], r['line']) for r in rows})
    for month, sensor, line in groups:
        selected = np.array([m and (r['month'], r['sensor'], r['line']) == (month, sensor, line)
                             for m, r in zip(mask, rows)])
        y = np.array([r['observed_mm'] for r in rows])[selected]
        p, old = predictions[selected], prior[selected]
        results.append({'month': month, 'sensor': sensor, 'line': line, 'n': int(selected.sum()),
                        'observed_mean_mm': float(y.mean()), 'predicted_mean_mm': float(p.mean()),
                        'rmse_mm': float(np.sqrt(np.mean((p-y)**2))),
                        'physics_only_rmse_mm': float(np.sqrt(np.mean((old-y)**2)))})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/joint_field_2026.json')
    parser.add_argument('--output', default='results/joint/field_2026')
    args = parser.parse_args()
    settings = json.loads((ROOT / args.config).read_bytes())
    physical = json.loads((ROOT / settings['physical_config']).read_bytes())
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = observations(settings, physical)
    development = np.array([row['split'] == 'development' for row in rows])
    folds = np.array([row['validation_fold'] for row in rows])
    y = np.array([row['observed_mm'] for row in rows])
    channels = [f"{r['month']}:{r['sensor']}:{r['line']}" for r in rows]
    A, target, obs, offset, neural = neural_problem(settings, physical, rows)
    problems = {'coupled_pinn': (A, target, obs, offset),
                'conventional_kerr': kerr_problem(settings, physical, rows, neural)}
    summaries, point_rows, selection_rows = {}, [], []
    solutions = {}
    for method, system in problems.items():
        objective = JointObjective(*system, y, channels)
        weight, selection = select_weight(objective, development, folds, settings['physics_weights'])
        evaluated = objective.fit(development, weight)
        updated = objective.fit(np.ones(len(rows), dtype=bool), weight)
        prediction, fitted = objective.predict(evaluated), objective.predict(updated)
        prior = objective.predict(neural['prior'] if method == 'coupled_pinn' else np.zeros_like(evaluated))
        solutions[method] = (evaluated, updated)
        summaries[method] = {'selected_physics_weight': weight,
            'development_n': int(development.sum()), 'test_n': int((~development).sum()),
            'test_rmse_mm': float(np.sqrt(np.mean((prediction[~development]-y[~development])**2))),
            'physics_only_test_rmse_mm': float(np.sqrt(np.mean((prior[~development]-y[~development])**2))),
            'test_groups': metrics(rows, prediction, prior, ~development),
            'all_data_fit_groups': metrics(rows, fitted, prior, np.ones(len(rows), dtype=bool)),
            'joint_physics_loss': objective.physics_loss(evaluated),
            'final_physics_loss': objective.physics_loss(updated)}
        point_rows.extend({'method': method, **row, 'physics_only_mm': float(p0),
                           'development_fit_mm': float(p), 'all_data_fit_mm': float(f)}
                          for row, p0, p, f in zip(rows, prior, prediction, fitted))
        selection_rows.extend({'method': method, **record} for record in selection)
        print(method, json.dumps(summaries[method]), flush=True)
    size = neural['size']
    evaluated, updated = solutions['coupled_pinn']
    for j, name in enumerate(STAGES):
        state, solver = neural['history'][j], neural['solvers'][j]
        q = updated[j*size:(j+1)*size]
        solver.solve(q)
        construction_load = state['load_offset'] + state['load_map'] @ updated
        for line, model in enumerate(solver.models):
            own = slice(line*solver.n, (line+1)*solver.n)
            other = slice((1-line)*solver.n, (2-line)*solver.n)
            model.load_correction = construction_load[own] - solver.operator[own, other] @ solver.coefficients[other]
        np.savez_compressed(out / f'{name}.npz', x_m=neural['sampled'],
            fields=neural['profiles'][j] @ q,
            development_fields=neural['profiles'][j] @ evaluated[j*size:(j+1)*size],
            physics_only_fields=neural['profiles'][j] @ neural['prior'][j*size:(j+1)*size],
            response_coordinates_kN=q, construction_load_kN=construction_load,
            coefficients=solver.coefficients,
            reference_extensions_m=state['reference_offset'] + state['reference_map'] @ updated,
            connection_extensions_m=state['connection_offset'] + state['connection_map'] @ updated,
            neural_output_coefficients=np.stack([response @ q for response in solver.output_response]))
        torch.save({'line_states': [m.state_dict() for m in solver.models],
                    'response_coordinates_kN': q, 'construction_load_kN': construction_load,
                    'scales': [m.scales for m in solver.models]}, out / f'{name}.pt')
    write_csv(out / 'point_predictions.csv', point_rows)
    write_csv(out / 'weight_selection.csv', selection_rows)
    write_json(out / 'assessment.json', {'evaluation': settings['evaluation'],
        'condition': settings['condition'], 'final_use': settings['final_use'], 'methods': summaries})
    write_json(out / 'config.json', settings)


if __name__ == '__main__':
    main()
