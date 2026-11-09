"""Condition a conventional Kerr beam on July and predict later excavation.

Only engineering configuration, casting records and original observations are
read. Full-ground operators, neural fields and recovery maps are not inputs.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from scipy.optimize import least_squares
from hangzhou_pinn.conventional_kerr import (
    KerrBeam, kerr_parameters, mindlin_vertical_stress, rectangular_stress,
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_loading(config, casting, grid, order):
    """Assemble pressures independently from physical construction records."""
    ground = config['ground']
    burial = ground['surface_height_m'] + ground['axis_below_base_m']
    poisson = ground['poisson_ratio']
    centres = ground['cavity_centres_y_m']
    wall_width = np.sqrt(3)*config['wall']['pile_diameter_m']/2
    xs, ys = config['x_divisions_m'], config['y_divisions_m']
    rectangles = {f'r{i}c{j}': [a+wall_width/2, b-wall_width/2,
                               c+wall_width/2, d-wall_width/2]
                  for i, (a, b) in enumerate(zip(xs[:-1], xs[1:]))
                  for j, (c, d) in enumerate(zip(ys[:-1], ys[1:]))}

    def rectangle_load(rectangle, depth, pressure):
        return np.stack([rectangular_stress(grid, np.full(len(grid), y), burial,
                         rectangle, depth, pressure, poisson, order) for y in centres])

    casting_pressures = []
    for group in casting['casting_groups']:
        c, d = np.asarray(group['road_chainage_m'])-casting['road_origin_chainage_m']
        force = -group['volume_m3']*casting['concrete_unit_weight_kN_m3']
        pressure = force/(len(xs)*wall_width*(d-c))
        casting_pressures.append(sum(rectangle_load([x-wall_width/2, x+wall_width/2, c, d],
                                                   0., pressure) for x in xs))
    casting_pressures = np.stack(casting_pressures)
    blinding = sum(casting_pressures[group['casting_group_index']]
                   * group['volume_m3']/casting['casting_groups'][group['casting_group_index']]['volume_m3']
                   for group in casting['blinding_groups'])
    first_gravity = casting_pressures.sum(axis=0)
    manual_extra = casting_pressures[2]*85/212

    pA, pB = config['active_pits']
    depths = config['depths_m']
    crown = {name: depths['crown'] for name in rectangles}
    a = {**crown, pA: depths['second_support']}
    ab = {**a, pB: depths['second_support']}
    lower_a = {**ab, pA: depths['final']}
    lower_ab = {**lower_a, pB: depths['final']}
    states = {'crown_cut': (crown, False, []), 'july': (crown, True, []),
              'cut_a': (a, True, []), 'support_a': (a, True, [pA]),
              'cut_ab': (ab, True, []), 'cut_ab_after_a': (ab, True, [pA]),
              'august': (ab, True, [pA, pB]), 'lower_a': (lower_a, True, [pA, pB]),
              'lower_ab': (lower_ab, True, [pA, pB]),
              'lower_a_without_second': (lower_a, True, []),
              'lower_ab_without_second': (lower_ab, True, [])}
    excavation = {}
    for name, rectangle in rectangles.items():
        for depth in sorted(set(depths.values())):
            excavation[name, depth] = rectangle_load(
                rectangle, depth, ground['effective_unit_weight_kN_m3']*depth)
    support_gravity = {}
    count = config['support']['members_per_cell']
    total_force = -config['support']['second_cast_volume_m3']*config['support']['unit_weight_kN_m3']
    point_force = total_force/(len(config['active_pits'])*count*2)
    source_depth = ground['surface_height_m']-config['support']['second_z_m']
    for pit in config['active_pits']:
        i, j = int(pit[1]), int(pit[3])
        endpoints = [(x, ys[j]+(ys[j+1]-ys[j])*(m+1)/(count+1))
                     for m in range(count) for x in (xs[i], xs[i+1])]
        support_gravity[pit] = np.stack([
            sum(point_force*mindlin_vertical_stress(grid-x, y-yp, burial, source_depth, poisson)
                for x, yp in endpoints) for y in centres])
    stage_pressures = {}
    for name, (depth_map, first, second) in states.items():
        pressure = blinding+sum(excavation[pit, depth] for pit, depth in depth_map.items())
        if first:
            pressure = pressure+first_gravity
        for pit in second:
            pressure = pressure+support_gravity[pit]
        stage_pressures[name] = pressure
    return first_gravity, manual_extra, stage_pressures, states, rectangles


def run(config_path, output):
    started = datetime.now(timezone.utc).isoformat()
    settings = json.loads(config_path.read_bytes())
    physical = json.loads((ROOT/settings['physical_config']).read_bytes())
    casting = json.loads((ROOT/physical['casting_loading_ref']).read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(-settings['half_length_m'], settings['half_length_m'],
                       int(round(2*settings['half_length_m']/settings['line_step_m']))+1)
    gravity, manual_extra, pressures, states, rectangles = source_loading(
        physical, casting, grid, settings['quadrature_order'])
    beam = physical['beam']
    diameter, inner = beam['outer_diameter_m'], beam['inner_diameter_m']
    B = beam['concrete_modulus_kPa']*np.pi*(diameter**4-inner**4)/64*beam['bending_reduction']
    S = (beam['concrete_modulus_kPa']/(2*(1+beam['poisson_ratio']))
         * np.pi*(diameter**2-inner**2)/4*beam['shear_reduction']*beam['shear_correction'])
    ground = physical['ground']
    burial = ground['surface_height_m']+ground['axis_below_base_m']
    with (ROOT/settings['calibration_observations']).open(encoding='utf-8-sig', newline='') as handle:
        july = [r for r in csv.DictReader(handle) if r['report_date'].startswith('2026-07')
                and r['interval_usable']=='True' and r['observable']=='track_bed_vertical_movement']
    automatic = [r for r in july if r['sensor']=='automated']
    if len(automatic) != 66:
        raise ValueError('The declared July calibration requires the original 66 automated points.')

    def solver_for(young):
        return KerrBeam(grid, B, S, diameter, young, ground['poisson_ratio'], burial)

    def predictions(solver, load, observations):
        result = np.zeros(len(observations))
        for line_index, line in enumerate(('up', 'down')):
            ids = [i for i, r in enumerate(observations) if r['line']==line]
            positions = np.array([float(observations[i]['chainage_m'])
                                  -physical['pit_origin_chainage_m'] for i in ids])
            solver.solve(load[line_index])
            result[ids] = solver.fields(positions)[:, 0]*1000
        return result

    observed = np.array([float(r['analysis_change_mm']) for r in automatic])
    fitted = least_squares(lambda log_E: predictions(solver_for(np.exp(log_E[0])), gravity,
                                                     automatic)-observed,
                           [np.log(1e5)], xtol=1e-10, ftol=1e-10, gtol=1e-10)
    if not fitted.success:
        raise RuntimeError(f'July effective-modulus fit failed: {fitted.message}')
    young = float(np.exp(fitted.x[0]))
    solver = solver_for(young)
    july_points, july_groups = [], []
    for sensor in ('automated', 'manual'):
        selected = [r for r in july if r['sensor']==sensor]
        prediction = predictions(solver, gravity+(manual_extra if sensor=='manual' else 0.), selected)
        for row, predicted in zip(selected, prediction):
            july_points.append({'sensor':sensor, 'line':row['line'], 'point_id':row['point_id'],
                'record_id':row['record_id'], 'previous_record_id':row['previous_record_id'],
                'observed_mm':float(row['analysis_change_mm']), 'predicted_mm':float(predicted)})
        for line in ('up', 'down'):
            rows = [r for r in july_points if r['sensor']==sensor and r['line']==line]
            p, o = np.array([r['predicted_mm'] for r in rows]), np.array([r['observed_mm'] for r in rows])
            july_groups.append({'sensor':sensor, 'line':line, 'n_points':len(rows),
                'observed_mean_mm':float(o.mean()), 'predicted_mean_mm':float(p.mean()),
                'rmse_mm':float(np.sqrt(np.mean((p-o)**2))), 'correlation':float(np.corrcoef(p,o)[0,1])})
    upper, lower, layer = kerr_parameters(young, ground['poisson_ratio'], burial)
    parameters = {'effective_modulus_MPa':young/1000, 'upper_kN_m3':upper, 'lower_kN_m3':lower,
                  'shear_kN_m':layer, 'burial_m':burial, 'poisson_ratio':ground['poisson_ratio'],
                  'diameter_m':diameter, 'bending_kN_m2':B, 'shear_rigidity_kN':S,
                  'conditioning':settings['conditioning'], 'parameter_source':settings['parameter_source']}
    # Persist the choice before reading any August observation target.
    write_json(output/'parameters.json', parameters)
    write_json(output/'calibration.json', {'n_automated':66, 'effective_modulus_MPa':young/1000,
        'automated_rmse_mm':float(np.sqrt(np.mean(fitted.fun**2))), 'channels':july_groups,
        'input_ref':settings['calibration_observations'], 'criterion':'Unweighted pointwise July automated MSE',
        'august_targets_used':False})
    write_csv(output/'july_point_predictions.csv', july_points)
    print(json.dumps({'parameters':parameters, 'july':july_groups}), flush=True)
    sampled = np.linspace(-settings['half_length_m'], settings['half_length_m'], 601)
    stage_fields, stage_summary = {}, []
    for name, pressure in pressures.items():
        fields = []
        for line in range(2):
            solver.solve(pressure[line])
            fields.append(solver.fields(sampled))
        fields = np.stack(fields)
        stage_fields[name] = fields
        np.savez_compressed(output/f'{name}.npz', x_m=sampled, fields=fields,
                            pressure_grid_m=grid, applied_pressure_kPa=pressure)
        volume = sum((rectangles[p][1]-rectangles[p][0])*(rectangles[p][3]-rectangles[p][2])*d
                     for p, d in states[name][0].items())
        for line in range(2):
            peak = np.argmax(fields[line,:,0])
            stage_summary.append({'stage':name, 'method':'conventional_kerr', 'line_index':line,
                'peak_uplift_mm':float(fields[line,peak,0]*1000), 'peak_x_m':float(sampled[peak]),
                'peak_abs_moment_kN_m':float(abs(fields[line,:,2]).max()), 'removed_volume_m3':volume})
    write_csv(output/'stage_summary.csv', stage_summary)
    with (ROOT/physical['observations']).open(encoding='utf-8-sig', newline='') as handle:
        observations = list(csv.DictReader(handle))
    points = []
    for row in observations:
        line = 0 if row['line']=='up' else 1
        x = float(row['chainage_m'])-physical['pit_origin_chainage_m']
        increment = stage_fields['august'][line,:,0]-stage_fields['july'][line,:,0]
        points.append({'method':'conventional_kerr', **row,
                       'predicted_increment_mm':float(np.interp(x, sampled, increment)*1000)})
    write_csv(output/'august_point_predictions.csv', points)
    write_json(output/'config.json', {'settings':settings, 'engineering_inputs':physical,
                                     'casting':casting, 'excavated_rectangles_m':rectangles})
    write_json(output/'run.json', {'started_at':started, 'finished_at':datetime.now(timezone.utc).isoformat(),
        'command':f'python tools/run_conventional_kerr.py --config {config_path.relative_to(ROOT).as_posix()}',
        'stages':list(states), 'model_scope':settings['model_scope'],
        'inputs':[settings['physical_config'], physical['casting_loading_ref'],
                  settings['calibration_observations'], physical['observations']],
        'baseline_kind':'conventional_kerr', 'parameters_ref':'parameters.json'})
    return parameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/conventional_kerr.json')
    parser.add_argument('--output', default='results/protection/field_2026/conventional_kerr')
    args = parser.parse_args()
    run(ROOT/args.config, ROOT/args.output)


if __name__=='__main__':
    main()
