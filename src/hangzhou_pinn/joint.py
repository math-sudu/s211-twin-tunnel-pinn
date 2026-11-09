"""Observation-conditioned neural fields with affine construction-state transfer."""

import numpy as np
from scipy.linalg import lstsq, qr


def history_system(stages, initial_references):
    """Assemble coupled residuals while forming supports in the learned state.

    Each stage uses neural response coordinates q in kN. The construction
    force remains load0 + L_e e, rather than being replaced by q. This
    distinction leaves a measurable physics residual in the joint objective.
    """
    size = stages[0]['motion_response'].shape[1]
    total = size * len(stages)
    reference_offset = np.asarray(initial_references, dtype=float).copy()
    reference_map = np.zeros((len(reference_offset), total))
    existing = set(stages[0]['active'])
    matrices, targets, states = [], [], []
    for index, stage in enumerate(stages):
        selection = np.eye(total)[index*size:(index+1)*size]
        active = np.asarray(stage['active'], dtype=int)
        born = sorted(set(active) - existing)
        if born:
            reference_offset[born] = connection_offset[born]
            reference_map[born] = connection_map[born]
        existing = set(active)
        load_offset = stage['load0'] + stage['reference_load_map'] @ reference_offset[active]
        load_map = stage['reference_load_map'] @ reference_map[active]
        matrices.append(stage['residual_matrix'] @ selection - stage['force_matrix'] @ load_map)
        targets.append(stage['force_matrix'] @ load_offset)
        motion_map = stage['motion_response'] @ selection
        connection_offset = (stage['connection_particular'] +
                             stage['reference_connection_map'] @ reference_offset[active])
        connection_map = (stage['reference_connection_map'] @ reference_map[active] +
                          stage['connection_response'] @ motion_map)
        states.append({'load_offset': load_offset, 'load_map': load_map,
                       'reference_offset': reference_offset.copy(),
                       'reference_map': reference_map.copy(),
                       'connection_offset': connection_offset.copy(),
                       'connection_map': connection_map.copy()})
    return (np.vstack(matrices) / np.sqrt(len(stages)),
            np.concatenate(targets) / np.sqrt(len(stages)), states)


class JointObjective:
    """One shared objective for data selection, fitting, and field evaluation.

    QR compresses the unchanged physics rows without altering their norm.
    Sensor/line/window channels receive equal total observation weight.
    """

    def __init__(self, physics_matrix, physics_target, observation_matrix,
                 observation_offset, observations, channels):
        self.column_scale = 1 / np.maximum(np.linalg.norm(physics_matrix, axis=0), 1e-15)
        joined = np.column_stack([physics_matrix * self.column_scale, physics_target])
        compressed = qr(joined, mode='r')[0]
        n = physics_matrix.shape[1]
        self.physics_matrix = compressed[:n+1, :n]
        self.physics_target = compressed[:n+1, n]
        self.observation_matrix = observation_matrix * self.column_scale
        self.offset = np.asarray(observation_offset)
        self.observations = np.asarray(observations)
        self.channels = np.asarray(channels)

    def fit(self, selected, physics_weight):
        selected = np.asarray(selected, dtype=bool)
        if physics_weight <= 0 or not np.any(selected):
            raise ValueError('Joint fitting requires positive physics weight and observations.')
        channels = self.channels[selected]
        unique, counts = np.unique(channels, return_counts=True)
        sizes = dict(zip(unique, counts))
        weights = np.array([1 / np.sqrt(len(unique) * sizes[c]) for c in channels])
        matrix = np.vstack([np.sqrt(physics_weight) * self.physics_matrix,
                            weights[:, None] * self.observation_matrix[selected]])
        target = np.r_[np.sqrt(physics_weight) * self.physics_target,
                       weights * (self.observations[selected] - self.offset[selected])]
        solution = lstsq(matrix, target, cond=1e-11, lapack_driver='gelsd')[0]
        return self.column_scale * solution

    def predict(self, solution):
        return self.offset + self.observation_matrix @ (solution / self.column_scale)

    def physics_loss(self, solution):
        residual = self.physics_matrix @ (solution / self.column_scale) - self.physics_target
        return float(residual @ residual)


def select_weight(objective, development, folds, candidates):
    """Choose the weight using only development-group validation targets."""
    development, folds = np.asarray(development), np.asarray(folds)
    records = []
    for weight in candidates:
        errors = []
        for fold in np.unique(folds[development]):
            training = development & (folds != fold)
            validation = development & (folds == fold)
            solution = objective.fit(training, weight)
            errors.extend((objective.predict(solution)[validation] -
                           objective.observations[validation])**2)
        records.append({'physics_weight': float(weight), 'validation_rmse_mm': float(np.sqrt(np.mean(errors)))})
    best = min(records, key=lambda row: row['validation_rmse_mm'])['physics_weight']
    return best, records
