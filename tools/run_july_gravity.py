"""Evaluate source-recorded concrete gravity before the August assessment."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1' if key=='OPENBLAS_NUM_THREADS' else '4'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from skfem import MeshWedge1
from hangzhou_pinn.foundation import CavityFoundation,cavity_mesh
from hangzhou_pinn.formation import FormedComponent
from hangzhou_pinn.excavation import excavate_mesh
from hangzhou_pinn.protection import BeamReference,connection_operators,casting_loads
from run_protection_stages import geometry,write_json,write_csv


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modulus-mpa',type=float,required=True)
    parser.add_argument('--output',default='results/protection/july_gravity')
    args=parser.parse_args()
    loads=json.loads((ROOT/'configs/july_concrete_loading.json').read_bytes())
    config=json.loads((ROOT/loads['model_config']).read_bytes())
    old=json.loads((ROOT/config['feature_configuration']).read_bytes())
    stations=np.asarray(old['source_config']['stations_m'])
    # Scale a single effective stiffness for the mature soil/treatment system;
    # concrete and lining stiffness retain their independent values.
    original_E=27.5
    config['ground']['unloading_multiplier']*=args.modulus_mpa/original_E
    ground,treatment,cover,walls,wall_ratio,members=geometry(config,stations)
    out=ROOT/args.output/f'E{args.modulus_mpa:g}'
    out.mkdir(parents=True,exist_ok=True)
    path=out/'gravity_response.npz'
    t=config['beam']
    area=np.pi*(t['outer_diameter_m']**2-t['inner_diameter_m']**2)/4
    inertia=np.pi*(t['outer_diameter_m']**4-t['inner_diameter_m']**4)/64
    B=t['concrete_modulus_kPa']*inertia*t['bending_reduction']
    S=t['concrete_modulus_kPa']/(2*(1+t['poisson_ratio']))*area*t['shear_reduction']*t['shear_correction']
    if path.exists():
        with np.load(path) as saved:
            C,ell=saved['operator'],saved['load']
    else:
        mesh_path=ROOT/'results/protection/parent_mesh.npz'
        if mesh_path.exists():
            with np.load(mesh_path) as saved:
                parent=MeshWedge1(saved['points'],saved['cells'])
        else:
            parent=cavity_mesh(ground,config['radius_m'])
            np.savez_compressed(mesh_path,points=parent.p,cells=parent.t)
        mesh,volume=excavate_mesh(parent,ground,{name:config['depths_m']['crown'] for name in ground['excavation_blocks_m']})
        f=CavityFoundation(mesh,{**ground,'removed_volume_m3':volume},config['radius_m'],stations)
        H,k,_=connection_operators(f.basis,members)
        active=[i for i,m in enumerate(members) if m['level']=='first']
        C,_,checks=f.condense(treatment,ground['reinforcement_multiplier'],
            additional_regions=[{'name':'weak_cover','boxes_m':cover,'modulus_multiplier':ground['reinforcement_multiplier']}],
            replacement_regions=[{'name':'walls','boxes_m':walls,'modulus_multiplier':wall_ratio}],
            formed_components=[FormedComponent(H[active],np.diag(k[active]),np.zeros(len(active)))])
        force=casting_loads(f.basis,config,walls,loads)
        ell=f.U.T@force
        np.savez_compressed(path,operator=C,load=ell,stations_m=stations,
                            resultant_kN=force[f.basis.nodal_dofs[2]].sum(axis=0))
        f.release_factor()
        write_json(out/'model.json',{'effective_modulus_MPa':args.modulus_mpa,'loads':loads,'ground':ground,'checks':checks})
    solver=BeamReference(stations,B,S,C)
    with (ROOT/'data/processed/monitoring/monthly_analysis.csv').open(encoding='utf-8-sig',newline='') as handle:
        observations=[r for r in csv.DictReader(handle) if r['report_date'].startswith('2026-07') and
                      r['interval_usable']=='True' and r['observable']=='track_bed_vertical_movement']
    points=[]
    # The June 27 blinding pour is before the automated reference but after
    # the manual reference. Its load uses the same representative wall segment.
    for sensor in ('automated','manual'):
        load=ell.sum(axis=1)
        if sensor=='manual':
            load=load+ell[:,2]*85/212
        solver.solve(load)
        for row in [r for r in observations if r['sensor']==sensor]:
            x=float(row['chainage_m'])-config['pit_origin_chainage_m']
            line=0 if row['line']=='up' else 1
            predicted=solver.fields(np.array([x]),line)[0,0]*1000
            points.append({'sensor':row['sensor'],'line':row['line'],'point_id':row['point_id'],
                           'record_id':row['record_id'],'previous_record_id':row['previous_record_id'],
                           'observed_mm':float(row['analysis_change_mm']),'predicted_mm':predicted})
    write_csv(out/'point_predictions.csv',points)
    summaries=[]
    for sensor,line in sorted({(r['sensor'],r['line']) for r in points}):
        group=[r for r in points if (r['sensor'],r['line'])==(sensor,line)]
        o=np.array([r['observed_mm'] for r in group]); p=np.array([r['predicted_mm'] for r in group])
        summaries.append({'sensor':sensor,'line':line,'observed_mean_mm':float(o.mean()),
                          'predicted_mean_mm':float(p.mean()),'rmse_mm':float(np.sqrt(np.mean((p-o)**2))),
                          'correlation':float(np.corrcoef(p,o)[0,1])})
    cal=[r for r in points if r['sensor']=='automated']
    loss=np.mean([(r['predicted_mm']-r['observed_mm'])**2 for r in cal])
    result={'effective_modulus_MPa':args.modulus_mpa,'automated_mse_mm2':float(loss),'channels':summaries}
    write_json(out/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
