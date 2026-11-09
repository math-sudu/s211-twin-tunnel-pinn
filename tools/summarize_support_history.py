"""Extract second-support formation and subsequent forces from saved states."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = json.loads((ROOT / 'configs/joint_field_2026.json').read_bytes())
    source = ROOT / settings['physical_results']
    water = ROOT / settings['water_results']
    results = ROOT / 'results/joint/field_2026'
    geometry = json.loads((source / 'model_geometry.json').read_bytes())
    with np.load(source / 'together_august_continuum_pinn.npz') as data:
        physical_reference = data['reference_extensions_m']
    with np.load(water / 'august.npz') as data:
        physical_reference += data['water_reference_extensions_m']
    with np.load(results / 'august.npz') as data:
        reference = data['reference_extensions_m']
        extensions = data['connection_extensions_m']

    rows = []
    for index, member in enumerate(geometry['members']):
        if member['level'] != 'second':
            continue
        ends = np.asarray(member['endpoints_m'])
        length = np.linalg.norm(ends[1] - ends[0])
        stiffness = member['young_kPa'] * member['area_m2'] / length
        deformation = extensions[index] - reference[index]
        rows.append({
            'member_id': member['id'],
            'length_m': float(length),
            'axial_stiffness_kN_per_m': float(stiffness),
            'physics_only_reference_mm': float(physical_reference[index] * 1000),
            'joint_reference_mm': float(reference[index] * 1000),
            'august_wall_extension_mm': float(extensions[index] * 1000),
            'post_installation_extension_mm': float(deformation * 1000),
            'august_axial_force_kN': float(stiffness * deformation),
        })

    output = results / 'support_history.csv'
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} second-support states to {output.relative_to(ROOT)}')
    for row in rows:
        print(f"{row['member_id']}: installation {row['joint_reference_mm']:.6f} mm, "
              f"subsequent extension {row['post_installation_extension_mm']:.6f} mm, "
              f"axial force {row['august_axial_force_kN']:.3f} kN")


if __name__ == '__main__':
    main()
