# Twin-tunnel response with construction history

Code for **Embedding construction history in a spatially coupled physics-informed neural network for twin-tunnel response to overlying excavation**.

The model couples continuous displacement, rotation, bending-moment and shear-force neural fields in two tunnels through a condensed three-dimensional ground model. Recovered wall movements define support-installation references. Joint fitting combines monitoring increments with equilibrium across construction states.

## Repository contents

- `src/hangzhou_pinn/`: ground and tunnel mechanics, regional neural fields, support formation, joint fitting and the Kerr comparator.
- `tools/`: scripts for feature training, staged analysis and result extraction.
- `pyproject.toml` and `requirements.txt`: package metadata and dependency versions.

This release contains the code. The authors will add the study data and input configurations separately. The field-analysis commands below require those input files.

## Installation

Use Python 3.11 or later. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m pip install -e .
```

The configured sparse solver is PARDISO, supplied by `pypardiso`. Ground assembly and factorization require sufficient memory for the three-dimensional mesh. The scripts use CPU computations and write generated results under `results/`.

## Required inputs

The study input files will use these paths:

- `configs/representative.json` and `configs/feature_stations.json`: auxiliary neural training and feature-transfer stations.
- `configs/field_protection_2026.json` and `configs/protection_stages.json`: ground, tunnel, wall and support geometry, materials and construction stages.
- `configs/july_concrete_loading.json`: concrete volumes, dates and loading intervals.
- `configs/conventional_kerr.json` and `configs/joint_field_2026.json`: comparator settings, observation paths, spatial groups and physics-weight selection.
- `data/processed/monitoring/monthly_analysis.csv`: baseline and target readings, dates, sensor systems, positions and displacement increments.
- `data/processed/engineering/august_2026_vertical_increment.csv`: July–August increments and model coordinates.

## Reproduction commands

After adding the input files at the paths above, run these commands in order:

```sh
python tools/prepare_hidden_features.py --stations-config configs/feature_stations.json
python tools/run_july_gravity.py --modulus-mpa 250
python tools/run_protection_stages.py
python tools/run_conventional_kerr.py
python tools/check_protection_water_boundary.py
python tools/run_joint_field.py
python tools/summarize_support_history.py
```

Feature preparation trains the auxiliary neural model and transfers its hidden layers to the tunnel regions. The staged ground analysis constructs the condensed operators and support-reference maps. The water-boundary calculation releases the initial pit-face water traction while retaining the specified pore-pressure field. Joint fitting selects the physics weight using the development groups, evaluates the reserved groups and assimilates all observations for structural estimation.

The main outputs are written under `results/joint/field_2026/`. They include pointwise displacement estimates, aggregate errors, physics-weight selection, structural fields and second-support force estimates. Units in the implementation are metres and kilonewtons; reported displacements are converted to millimetres.
