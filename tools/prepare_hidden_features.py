"""Train the auxiliary neural features and transfer them to the ground stations."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hangzhou_pinn.ground import GroundModel, beam_and_foundation
from hangzhou_pinn.mixed import MixedPINN, dde, torch


def transfer_hidden_features(weights, stations, pit_half_width, layers):
    transferred = {}
    for i, (left, right) in enumerate(zip(stations[:-1], stations[1:])):
        midpoint = (left + right) / 2
        source = 0 if midpoint < -pit_half_width else 2 if midpoint > pit_half_width else 1
        for layer in range(layers):
            for parameter in ('weight', 'bias'):
                transferred[f'nets.{i}.linears.{layer}.{parameter}'] = weights[
                    f'nets.{source}.linears.{layer}.{parameter}'].clone()
    return transferred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/representative.json')
    parser.add_argument('--stations-config', required=True)
    parser.add_argument('--output', default='results/features/hidden.pt')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_bytes())
    stations = json.loads((ROOT / args.stations_config).read_bytes())['source_config']['stations_m']
    training = config['pinn']
    dde.config.set_random_seed(training['seed'])
    case = next(item for item in config['cases'] if item['name'] == 'reinforced_full')
    ground = GroundModel(config['ground'], case['reinforcement']).solve_stage(case['blocks'])
    coefficients = beam_and_foundation(config)
    half_width = config['ground']['excavation_width_m'] / 2
    edges = [training['domain_m'][0], -half_width, half_width, training['domain_m'][1]]
    model = MixedPINN(edges, [coefficients.copy() for _ in range(3)],
                      ground['x_m'], ground['g_m'], training['width'], training['layers'])
    model.train_physics(training)
    weights = transfer_hidden_features(model.state_dict(), stations, half_width, training['layers'])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, output)
    print(f'Saved hidden neural features to {output}')


if __name__ == '__main__':
    main()
