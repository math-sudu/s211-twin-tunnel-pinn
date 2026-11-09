"""Run source-constrained local excavation, wall formation and later stages."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1' if variable=='OPENBLAS_NUM_THREADS' else '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy.sparse import diags
from skfem import MeshWedge1
from hangzhou_pinn.excavation import excavate_mesh, excavation_surface_load
from hangzhou_pinn.formation import FormedComponent
from hangzhou_pinn.foundation import CavityFoundation, cavity_mesh
from hangzhou_pinn.ground import unloading_modulus
from hangzhou_pinn.mixed import torch, dde
from hangzhou_pinn.protection import (BeamReference, CoupledPINN, connection_operators,
                                      casting_loads)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry(config, stations):
    ground = dict(config['ground'])
    xs, ys, wall = config['x_divisions_m'], config['y_divisions_m'], config['wall']
    d, spacing = wall['pile_diameter_m'], wall['pile_spacing_m']
    area, inertia = np.pi*d*d/4, np.pi*d**4/64
    thickness = np.sqrt(12*inertia/area)
    wall_E = wall['concrete_young_kPa']*area/(spacing*thickness)
    bottom, top = ground['surface_height_m']-wall['pile_length_m'], ground['surface_height_m']
    walls = [[x-thickness/2, x+thickness/2, ys[0]-thickness/2, ys[-1]+thickness/2, bottom, top] for x in xs]
    walls += [[xs[0]-thickness/2, xs[-1]+thickness/2, y-thickness/2, y+thickness/2, bottom, top] for y in ys]
    blocks = {f'r{i}c{j}': [a,b,c,d] for i,(a,b) in enumerate(zip(xs[:-1],xs[1:]))
              for j,(c,d) in enumerate(zip(ys[:-1],ys[1:]))}
    roof = [[xs[0], xs[-1], ys[0], ys[-1], -3., 0.]]
    legs = [[xs[0], xs[-1], a, b, -14.2, -3.] for a,b in config['portal_legs_y_m']]
    cover = [[xs[0], xs[-1], ys[0], ys[-1], 0., top]]
    ground.update(excavation_blocks_m=blocks, retained_wall_boxes_m=walls,
                  longitudinal_anchors_m=sorted(set([*stations, *xs,
                     *[v for x in xs for v in (x-thickness/2,x+thickness/2)]])),
                  cross_section_partitions_m=[box[2:] for box in walls+legs] +
                      [[ys[0]-thickness/2, ys[-1]+thickness/2, bottom, 0.]])
    E = unloading_modulus(ground['compression_modulus_kPa'], ground['poisson_ratio'], ground['unloading_multiplier'])
    members = []
    support = config['support']
    for level in ('first', 'second'):
        for name, (a,b,c,d) in blocks.items():
            if level == 'second' and name not in config['active_pits']:
                continue
            for j in range(support['members_per_cell']):
                y = c+(d-c)*(j+1)/(support['members_per_cell']+1)
                members.append({'id': f'{level}_{name}_{j}', 'level': level, 'pit': name,
                    'endpoints_m': [[a,y,support[f'{level}_z_m']], [b,y,support[f'{level}_z_m']]],
                    'young_kPa':support['young_kPa'], 'area_m2':support[f'{level}_area_m2'],
                    'unit_weight_kN_m3':support['unit_weight_kN_m3']})
    return ground, roof+legs, cover, walls, wall_E/E, members


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/field_protection_2026.json')
    parser.add_argument('--output', default='results/protection/field_2026')
    parser.add_argument('--through', choices=['july', 'august', 'all'], default='all')
    parser.add_argument('--refresh-support-weight', action='store_true',
                        help='Update only documented support gravity using existing stage operators.')
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    config = json.loads((ROOT/args.config).read_bytes())
    casting = (json.loads((ROOT/config['casting_loading_ref']).read_bytes())
               if 'casting_loading_ref' in config else None)
    if casting:
        config['resolved_casting_loads'] = casting
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists():
        previous = json.loads((out/'config.json').read_bytes())
        def without_support_weight(c):
            c=copy.deepcopy(c)
            for key in ('second_cast_volume_m3','second_cast_record_id'):
                c['support'].pop(key,None)
            return c
        unchanged = (without_support_weight(previous)==without_support_weight(config)
                     if args.refresh_support_weight else previous==config)
        if not unchanged:
            raise ValueError('Output contains a different physical configuration; choose a different output directory.')
    write_json(out/'config.json', config)
    old = json.loads((ROOT/config['feature_configuration']).read_bytes())
    stations = np.asarray(old['source_config']['stations_m'])
    ground, treatment, cover, walls, wall_ratio, members = geometry(config, stations)
    write_json(out/'model_geometry.json', {'ground':ground, 'treatment_boxes_m':treatment,
               'cover_boxes_m':cover, 'walls':walls, 'members':members,
               'wall_modulus_multiplier':wall_ratio, 'stations_m':stations.tolist()})
    t = config['beam']
    area = np.pi*(t['outer_diameter_m']**2-t['inner_diameter_m']**2)/4
    inertia = np.pi*(t['outer_diameter_m']**4-t['inner_diameter_m']**4)/64
    B = t['concrete_modulus_kPa']*inertia*t['bending_reduction']
    S = t['concrete_modulus_kPa']/(2*(1+t['poisson_ratio']))*area*t['shear_reduction']*t['shear_correction']
    checkpoint = torch.load(ROOT/config['feature_checkpoint'], weights_only=True)
    dde.config.set_random_seed(20261029)
    parent = None
    first = [i for i,m in enumerate(members) if m['level']=='first']
    second = {pit:[i for i,m in enumerate(members) if m['level']=='second' and m['pit']==pit]
              for pit in config['active_pits']}
    pA, pB = config['active_pits']
    all_second = second[pA]+second[pB]
    ideal_second_volume=sum(members[i]['area_m2']*np.linalg.norm(
        np.subtract(*members[i]['endpoints_m'])) for i in all_second)
    weight_factor=config['support'].get('second_cast_volume_m3',ideal_second_volume)/ideal_second_volume
    d = config['depths_m']
    depth0 = {name:d['crown'] for name in ground['excavation_blocks_m']}
    depthA = {**depth0,pA:d['second_support']}
    depthAB = {**depthA,pB:d['second_support']}
    lowerA = {**depthAB,pA:d['final']}
    lowerAB = {**lowerA,pB:d['final']}
    stages = {'crown_cut':(depth0,[]), 'july':(depth0,first),
              'cut_a':(depthA,first), 'support_a':(depthA,first+second[pA]),
              'cut_ab':(depthAB,first), 'cut_ab_after_a':(depthAB,first+second[pA]),
              'august':(depthAB,first+all_second),
              'lower_a':(lowerA,first+all_second), 'lower_ab':(lowerAB,first+all_second),
              'lower_a_without_second':(lowerA,first), 'lower_ab_without_second':(lowerAB,first)}
    def stage_data(name):
        nonlocal parent
        path = out/f'{name}_condensed.npz'
        cached = None
        if path.exists():
            with np.load(path) as data:
                cached = {key:data[key] for key in data.files}
            old_factor=float(cached.get('second_support_weight_factor',1.))
            if old_factor!=weight_factor and not args.refresh_support_weight:
                raise ValueError('Support gravity changed; finish the explicit --refresh-support-weight update.')
            if old_factor==weight_factor:
                return cached
            if not set(cached['active']).intersection(all_second):
                cached['second_support_weight_factor']=weight_factor
                np.savez_compressed(path,**cached)
                return cached
        started = perf_counter()
        if parent is None:
            mesh_path=ROOT/config.get('parent_mesh_ref',str(out/'parent_mesh.npz'))
            if mesh_path.exists():
                with np.load(mesh_path) as saved:
                    parent = MeshWedge1(saved['points'],saved['cells'])
            else:
                parent = cavity_mesh(ground,config['radius_m'])
                np.savez_compressed(mesh_path,points=parent.p,cells=parent.t)
        depths, active = stages[name]
        mesh, volume = excavate_mesh(parent,ground,depths)
        print(f'Assembling {name}: {mesh.nelements} wedges',flush=True)
        foundation = CavityFoundation(mesh,{**ground,'removed_volume_m3':volume},config['radius_m'],stations)
        H,k,weights = connection_operators(foundation.basis,members)
        formed = ([FormedComponent(H[active],np.diag(k[active]),np.zeros(len(active)))] if active else [])
        C,_,checks = foundation.condense(treatment,ground['reinforcement_multiplier'],
            additional_regions=[{'name':'weak_cover','boxes_m':cover,'modulus_multiplier':ground['reinforcement_multiplier']}],
            replacement_regions=[{'name':'retaining_walls','boxes_m':walls,'modulus_multiplier':wall_ratio}],
            formed_components=formed,load_only=cached is not None)
        if cached is not None:
            support_ids=[i for i in active if i in all_second]
            change=np.asarray(weights[support_ids].sum(axis=0)).ravel()*(weight_factor-old_factor)
            delta_load,delta_u,delta_checks=foundation.condense_loading(change,np.zeros_like(change))
            cached['load0']+=delta_load
            cached['connection_particular']+=np.asarray(H@delta_u)
            cached['second_support_weight_factor']=weight_factor
            np.savez_compressed(path,**cached)
            record=json.loads((out/f'{name}_assembly.json').read_bytes())
            record['loading']['applied_vertical_force_kN']+=delta_checks['applied_vertical_force_kN']
            record['support_gravity']={'total_cast_volume_m3':config['support']['second_cast_volume_m3'],
                'record_id':config['support']['second_cast_record_id'],
                'idealized_member_volume_m3':ideal_second_volume,
                'load_update_seconds':perf_counter()-started}
            write_json(out/f'{name}_assembly.json',record)
            foundation.release_factor()
            return cached
        force = excavation_surface_load(foundation.basis,ground,config['lateral_stress_ratio'])
        if casting:
            gravity = casting_loads(foundation.basis,config,walls,casting)
            for group in casting['blinding_groups']:
                i = group['casting_group_index']
                force += gravity[:,i]*group['volume_m3']/casting['casting_groups'][i]['volume_m3']
            if any(i in active for i in first):
                force += gravity.sum(axis=1)
            added_weight = [i for i in active if members[i]['level']=='second']
        else:
            added_weight = active
        if len(added_weight):
            for i in added_weight:
                force += np.asarray(weights[i].toarray()).ravel()*(weight_factor if i in all_second else 1.)
        ell,u0,load_checks = foundation.condense_loading(force,np.zeros_like(force))
        rhs = H[active].T @ diags(k[active])
        response = foundation.factor.solve((foundation.Z.T @ rhs).toarray())
        data = {'operator':C, 'load0':ell, 'reference_load_map':np.asarray(foundation.U.T @ rhs),
                'connection_particular':np.asarray(H @ u0),
                'reference_connection_map':np.asarray((H @ foundation.Z) @ response),
                'connection_response':np.asarray(H @ foundation.U), 'active':np.asarray(active,dtype=int),
                'second_support_weight_factor':weight_factor,
                'removed_volume_m3':volume, 'assembly_seconds':perf_counter()-started}
        np.savez_compressed(path,**data)
        write_json(out/f'{name}_assembly.json',{'operator':checks,'loading':load_checks,
                   'assembly_seconds':float(data['assembly_seconds']), 'depths_m':depths,
                   'active_members':[members[i]['id'] for i in active]})
        foundation.release_factor()
        del foundation,response
        gc.collect()
        return data

    sequences = {'together':['crown_cut','july','cut_ab','august','lower_a','lower_ab'],
                 'sequential':['crown_cut','july','cut_a','support_a','cut_ab_after_a','august','lower_a','lower_ab'],
                 'without_second':['crown_cut','july','cut_ab','lower_a_without_second','lower_ab_without_second']}
    if args.through == 'july':
        sequences = {'together':['crown_cut','july']}
    elif args.through == 'august':
        sequences = {'together':sequences['together'][:4]}
    summary = []
    diagnostics = json.loads((out/'solver_records.json').read_bytes()) if (out/'solver_records.json').exists() else {}
    sampled = np.linspace(stations[0],stations[-1],601)
    solvers = {}
    saved_fields = {}
    for sequence,names in sequences.items():
        references = {method:np.zeros(len(members)) for method in ('continuum_fe','continuum_pinn')}
        existing = set()
        connection_states = {}
        for name in names:
            data = stage_data(name)
            active = data['active']
            born = set(active)-existing
            for method in references:
                if born:
                    ids = sorted(born)
                    references[method][ids] = connection_states[method][ids]
            existing = set(active)
            reused = (not args.refresh_support_weight and
                      all((out/f'{sequence}_{name}_{method}.npz').exists() for method in references))
            if name not in solvers and not reused:
                start = perf_counter()
                C = data['operator']
                solvers[name] = {'continuum_fe':BeamReference(stations,B,S,C),
                    'continuum_pinn':CoupledPINN(stations,config['pinn_edges_m'],B,S,C,checkpoint,stations)}
                diagnostics[name] = {'solver_setup_seconds':perf_counter()-start,
                                     'pinn':solvers[name]['continuum_pinn'].records}
                write_json(out/'solver_records.json',diagnostics)
            for method in references:
                r = references[method][active]
                if reused:
                    with np.load(out/f'{sequence}_{name}_{method}.npz') as saved:
                        a,fields = saved['coefficients'],saved['fields']
                else:
                    solver = solvers[name][method]
                    load = data['load0']+data['reference_load_map']@r
                    a = solver.solve(load)
                    fields = np.stack([solver.fields(sampled,line) for line in range(2)])
                    np.savez_compressed(out/f'{sequence}_{name}_{method}.npz',x_m=sampled,fields=fields,
                                        coefficients=a,reference_extensions_m=references[method],load_kN=load)
                connection_states[method] = (data['connection_particular']+data['reference_connection_map']@r+
                                             data['connection_response']@a)
                saved_fields[sequence,name,method] = fields.copy()
                for line in range(2):
                    f = fields[line]
                    peak = int(np.argmax(f[:,0]))
                    summary.append({'sequence':sequence,'stage':name,'method':method,'line_index':line,
                        'peak_uplift_mm':float(f[peak,0]*1000),'peak_x_m':float(sampled[peak]),
                        'peak_abs_moment_kN_m':float(np.max(abs(f[:,2]))),
                        'removed_volume_m3':float(data['removed_volume_m3'])})
                print(sequence,name,method,json.dumps(summary[-2:]),flush=True)
            write_csv(out/'stage_summary.csv',summary)
            # Reusing the soil/solver matrices is safe; formation references are
            # independently advanced for each chronological path and method.
    observation_rows=[]
    if ('together','august','continuum_pinn') in saved_fields:
        with (ROOT/config['observations']).open(encoding='utf-8-sig',newline='') as f:
            observations=list(csv.DictReader(f))
        for method in references:
            after=saved_fields['together','august',method]
            before=saved_fields['together','july',method]
            for row in observations:
                line=0 if row['line']=='up' else 1
                x=float(row['chainage_m'])-config['pit_origin_chainage_m']
                prediction=np.interp(x,sampled,after[line,:,0]-before[line,:,0])*1000
                observation_rows.append({'method':method,**row,'predicted_increment_mm':float(prediction)})
        write_csv(out/'august_point_predictions.csv',observation_rows)
    write_json(out/'run.json',{'started_at':started_at,'finished_at':datetime.now(timezone.utc).isoformat(),
               'command':'python '+' '.join(sys.argv),'through':args.through,
               'sequences':sequences,'conditions':config['conditions']})


if __name__=='__main__':
    main()
