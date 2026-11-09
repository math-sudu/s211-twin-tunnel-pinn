"""Evaluate removal of pit water pressure at unchanged retained pore pressure.

The existing excavation load releases effective stress at hydrostatic head
z=9.7 m. This calculation adds removal of that boundary water pressure. It
does not prescribe a drawdown in the retained ground or infer an actual head
history from the monitoring observations.
"""

import copy
import csv
import json
from run_protection_stages import (
    ROOT, np, MeshWedge1, CavityFoundation, FormedComponent, BeamReference,
    geometry, connection_operators, excavate_mesh, excavation_surface_load,
    write_json, write_csv,
)
from run_conventional_kerr import source_loading
from hangzhou_pinn.conventional_kerr import KerrBeam


def main():
    source = ROOT/'results/protection/field_2026'
    out = ROOT/'results/protection/water_boundary'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((source/'config.json').read_bytes())
    stations = np.array(json.loads((source/'model_geometry.json').read_bytes())['stations_m'])
    ground, treatment, cover, walls, wall_ratio, members = geometry(config, stations)
    with np.load(ROOT/config['parent_mesh_ref']) as saved:
        parent = MeshWedge1(saved['points'], saved['cells'])
    t = config['beam']
    area = np.pi*(t['outer_diameter_m']**2-t['inner_diameter_m']**2)/4
    inertia = np.pi*(t['outer_diameter_m']**4-t['inner_diameter_m']**4)/64
    B = t['concrete_modulus_kPa']*inertia*t['bending_reduction']
    S = t['concrete_modulus_kPa']/(2*(1+t['poisson_ratio']))*area*t['shear_reduction']*t['shear_correction']
    sequence = ['crown_cut', 'july', 'cut_ab', 'august', 'lower_a', 'lower_ab']
    water_references = np.zeros(len(members))
    existing = set()
    fields, x = {}, np.linspace(-150, 150, 601)
    for stage in sequence:
        with np.load(source/f'{stage}_condensed.npz') as saved:
            data = {key: saved[key] for key in saved.files}
        active = data['active']
        born = sorted(set(active)-existing)
        if born:
            water_references[born] = connection[born]
        existing = set(active)
        cache = out/f'{stage}_water_load.npz'
        if not cache.exists():
            depths = json.loads((source/f'{stage}_assembly.json').read_bytes())['depths_m']
            mesh, volume = excavate_mesh(parent, ground, depths)
            foundation = CavityFoundation(mesh, {**ground, 'removed_volume_m3': volume}, config['radius_m'], stations)
            H, k, _ = connection_operators(foundation.basis, members)
            formed = [FormedComponent(H[active], np.diag(k[active]), np.zeros(len(active)))] if len(active) else []
            foundation.condense(treatment, ground['reinforcement_multiplier'],
                additional_regions=[{'name': 'cover', 'boxes_m': cover, 'modulus_multiplier': ground['reinforcement_multiplier']}],
                replacement_regions=[{'name': 'walls', 'boxes_m': walls, 'modulus_multiplier': wall_ratio}],
                formed_components=formed, load_only=True)
            pressure = excavation_surface_load(foundation.basis,
                {**ground, 'effective_unit_weight_kN_m3': 9.81}, 1.)
            ell, u0, _ = foundation.condense_loading(pressure, np.zeros_like(pressure))
            np.savez_compressed(cache, load=ell, connection_particular=np.asarray(H@u0))
            foundation.release_factor()
            del foundation
        with np.load(cache) as saved:
            water_load, particular = saved['load'], saved['connection_particular']
        r = water_references[active]
        load = water_load+data['reference_load_map']@r
        solver = BeamReference(stations, B, S, data['operator'], max_step=.5)
        a = solver.solve(load)
        delta = np.stack([solver.fields(x, line) for line in range(2)])
        connection = particular+data['reference_connection_map']@r+data['connection_response']@a
        with np.load(source/f'together_{stage}_continuum_pinn.npz') as saved:
            original = saved['fields']
        fields[stage] = original+delta
        np.savez_compressed(out/f'{stage}.npz', x_m=x, fields=fields[stage],
                            water_release_fields=delta, water_reference_extensions_m=water_references)
        print(stage, 'peak added water-release displacement (mm)', float(delta[:,:,0].max()*1000), flush=True)

    # The conventional comparator receives the same added bottom pressure.
    # Its already selected July modulus is retained; no observations are fitted.
    settings = json.loads((ROOT/'configs/conventional_kerr.json').read_bytes())
    casting = json.loads((ROOT/config['casting_loading_ref']).read_bytes())
    physical = copy.deepcopy(config)
    physical['ground']['effective_unit_weight_kN_m3'] += 9.81
    grid = np.arange(-150., 150.25, .5)
    _, _, pressures, _, _ = source_loading(physical, casting, grid, settings['quadrature_order'])
    parameters = json.loads((source/'conventional_kerr/parameters.json').read_bytes())
    kerr = KerrBeam(grid, B, S, config['beam']['outer_diameter_m'],
                    parameters['effective_modulus_MPa']*1000, ground['poisson_ratio'], 17.8)
    kerr_fields = {}
    for stage in sequence:
        value = []
        for line in range(2):
            kerr.solve(pressures[stage][line])
            value.append(kerr.fields(x))
        kerr_fields[stage] = np.stack(value)
    with (ROOT/config['observations']).open(encoding='utf-8-sig', newline='') as handle:
        observations = list(csv.DictReader(handle))
    points, groups, future = [], [], []
    for method, state in [('coupled', fields), ('conventional_kerr', kerr_fields)]:
        for row in observations:
            line = 0 if row['line']=='up' else 1
            pos = float(row['chainage_m'])-config['pit_origin_chainage_m']
            prediction = np.interp(pos, x, state['august'][line,:,0]-state['july'][line,:,0])*1000
            points.append({'method': method, **row, 'predicted_increment_mm': float(prediction)})
        for sensor in ('automated', 'manual'):
            for line in ('up', 'down'):
                selected = [r for r in points if (r['method'],r['sensor'],r['line'])==(method,sensor,line)]
                p = np.array([r['predicted_increment_mm'] for r in selected])
                o = np.array([float(r['observed_increment_mm']) for r in selected])
                groups.append({'method': method, 'sensor': sensor, 'line': line,
                    'predicted_mean_mm': float(p.mean()), 'observed_mean_mm': float(o.mean()),
                    'rmse_mm': float(np.sqrt(np.mean((p-o)**2)))})
    for stage in ('lower_a', 'lower_ab'):
        for row in observations:
            line = 0 if row['line']=='up' else 1
            pos = float(row['chainage_m'])-config['pit_origin_chainage_m']
            inc = np.interp(pos,x,fields[stage][line,:,0]-fields['august'][line,:,0])*1000
            future.append({'scenario':stage, 'sensor':row['sensor'], 'line':row['line'],
                'point_id':row['point_id'], 'chainage_m':row['chainage_m'], 'reference_date':row['target_date'],
                'reference_observed_cumulative_mm':row['target_cumulative_mm'],
                'predicted_additional_movement_mm':float(inc),
                'conditional_cumulative_mm':float(row['target_cumulative_mm'])+float(inc)})
    write_csv(out/'august_point_predictions.csv', points)
    write_csv(out/'later_stage_predictions.csv', future)
    write_json(out/'assessment.json', {'condition': 'Initial hydrostatic head at original ground surface; removal of pit boundary water pressure in addition to effective-stress release; retained pore-pressure field unchanged. No inferred actual groundwater history.',
        'water_unit_weight_kN_m3':9.81, 'field_comparison': groups,
        'solver': 'Existing mixed PINN fields plus the linear water-load response from a 0.5 m Timoshenko beam on the identical condensed ground operator; support references propagated for this load component.'})
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
