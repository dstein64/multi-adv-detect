import argparse
from collections import namedtuple
import csv
from itertools import count, takewhile
import multiprocessing
import os
import random
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from utils import ATTACKS, DEFAULT_WORKSPACE, NUM_MODELS, REPRESENTATIONS_SIZE, set_seed


LOCK = multiprocessing.Lock()

run_idx = multiprocessing.Value('i', 0)

Params = namedtuple('Params', [
    'seed',
    'num_units',
    'trial_idx',
    'workspace',
    'eval_correct',
    'train_proportion',
    'root_outdir',
])


def run(params):
    with LOCK:
        print('run_idx:', run_idx.value, 'num_units:', params.num_units, 'trial_idx:', params.trial_idx)
        run_idx.value += 1
    set_seed(params.seed)
    with LOCK:
        # Randomly draw a set of models.
        models = sorted(list(np.random.choice(range(1, NUM_MODELS + 1), size=params.num_units, replace=False)))
        # Randomly draw a unit for each model.
        units = list(np.random.choice(range(REPRESENTATIONS_SIZE), size=params.num_units, replace=True))
        representations = []
        adv_repr_lookup = {attack: [] for attack in ATTACKS}
        # WARN: The data in representations.hdf5 is transposed for faster loading.
        for model, unit in zip(models, units):
            repr_path = os.path.join(params.workspace, 'eval', 'by_model', str(model), 'representations.hdf5')
            with h5py.File(repr_path, 'r') as f:
                repr_subset = f['representations']
                assert repr_subset.shape[0] == REPRESENTATIONS_SIZE
                repr_subset = repr_subset[unit]
                # Limit to images that were correctly classified initially.
                # This was already done earlier in the pipeline for the adversarial images.
                # This is not included as part of the indexing above, as it's faster this way.
                repr_subset = repr_subset[params.eval_correct, np.newaxis]
                assert type(repr_subset) == np.ndarray
            representations.append(repr_subset)
            for attack in ATTACKS:
                adv_repr_path = os.path.join(
                    params.workspace, 'attack', attack, 'by_model', str(model), 'representations.hdf5')
                with h5py.File(adv_repr_path, 'r') as f:
                    adv_repr_subset = f['representations']
                    assert adv_repr_subset.shape[0] == REPRESENTATIONS_SIZE
                    adv_repr_subset = adv_repr_subset[unit][:, np.newaxis]
                    assert repr_subset.shape == adv_repr_subset.shape
                    assert type(adv_repr_subset) == np.ndarray
                adv_repr_lookup[attack].append(adv_repr_subset)
        representations = np.hstack(representations)
        adv_repr_lookup = {
            attack: np.hstack(adv_repr_subset) for attack, adv_repr_subset in adv_repr_lookup.items()
        }
    for adv_source in ATTACKS:
        idxs = np.arange(representations.shape[0])
        np.random.shuffle(idxs)
        train_size = int(params.train_proportion * len(idxs))
        test_size = len(idxs) - train_size
        train_idxs = idxs[0:train_size]
        test_idxs = idxs[train_size:]
        adv_source_repr = adv_repr_lookup[adv_source]
        X_train = np.concatenate((representations[train_idxs], adv_source_repr[train_idxs]))
        y_train = np.concatenate((np.zeros(train_size, dtype=int), np.ones(train_size, dtype=int)))
        clf = MLPClassifier()
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
            clf.fit(X_train, y_train)
        for adv_target in ATTACKS:
            adv_target_repr = adv_repr_lookup[adv_target]
            X_test = np.concatenate((representations[test_idxs], adv_target_repr[test_idxs]))
            y_test = np.concatenate((np.zeros(test_size, dtype=int), np.ones(test_size, dtype=int)))
            y_proba = clf.predict_proba(X_test)
            y_pred = y_proba.argmax(axis=1)
            correct = y_pred == y_test
            accuracy = sum(correct) / len(correct)
            outdir = os.path.join(params.root_outdir, adv_source, adv_target)
            csv_path = os.path.join(outdir, f'{params.num_units}.csv')
            with LOCK, open(csv_path, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    ','.join(str(x) for x in models),  # models
                    ','.join(str(x) for x in units),   # units
                    sum(correct),                      # correct
                    accuracy                           # accuracy
                ])


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument('--seed', type=int, default=0)
    default_num_units_domain = list(takewhile(lambda x: x <= NUM_MODELS, (2 ** x for x in count(3))))
    parser.add_argument('--num-units-domain', nargs='+', type=int, default=default_num_units_domain)
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--train-proportion', type=float, default=0.90)
    parser.add_argument('--num-jobs', type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    set_seed(args.seed)
    attacked_model = 0
    eval_correct_path = os.path.join(
        args.workspace, 'eval', 'by_model', str(attacked_model), 'correct.csv')
    eval_correct = np.loadtxt(eval_correct_path, dtype=bool, delimiter=',')
    root_outdir = os.path.join(args.workspace, 'detect_unit_wise_treatment')
    os.makedirs(root_outdir, exist_ok=True)
    for adv_source in ATTACKS:
        for adv_target in ATTACKS:
            outdir = os.path.join(root_outdir, adv_source, adv_target)
            os.makedirs(outdir, exist_ok=True)
            for num_units in args.num_units_domain:
                with open(os.path.join(outdir, f'{num_units}.csv'), 'w') as f:
                    f.write('models,units,correct,accuracy\n')
    params_list = []
    for num_units in args.num_units_domain:
        for trial_idx in range(args.num_trials):
            params = Params(
                seed=args.seed + len(params_list),
                num_units=num_units,
                trial_idx=trial_idx,
                workspace=args.workspace,
                eval_correct=eval_correct,
                train_proportion=args.train_proportion,
                root_outdir=root_outdir
            )
            params_list.append(params)
    # Shuffle parameters list to stagger data reading to improve utilization. Without shuffling,
    # consecutive trials have the same amount of data to read, so reads would be more likely to
    # occur in parallel.
    random.shuffle(params_list)
    print(f'Running: {len(params_list)} runs')
    with multiprocessing.Pool(processes=args.num_jobs) as pool:
        pool.map_async(run, params_list)
        pool.close()
        pool.join()
    # Aggregate
    print('Aggregate')
    aggregated_dict = {
        'adv_source': [],
        'adv_target': [],
        'num_units': [],
        'accuracy_mean': [],
        'accuracy_std': [],
        'accuracy_count': []
    }
    for adv_source in ATTACKS:
        for adv_target in ATTACKS:
            for num_units in args.num_units_domain:
                df = pd.read_csv(os.path.join(root_outdir, adv_source, adv_target, f'{num_units}.csv'))
                aggregated_dict['adv_source'].append(adv_source)
                aggregated_dict['adv_target'].append(adv_target)
                aggregated_dict['num_units'].append(num_units)
                aggregated_dict['accuracy_mean'].append(df.accuracy.mean())
                aggregated_dict['accuracy_std'].append(df.accuracy.std(ddof=1))
                aggregated_dict['accuracy_count'].append(df.accuracy.count())
    aggregated_df = pd.DataFrame.from_dict(aggregated_dict)
    print(aggregated_df)
    aggregated_path = os.path.join(root_outdir, 'aggregated.csv')
    aggregated_df.to_csv(aggregated_path, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
