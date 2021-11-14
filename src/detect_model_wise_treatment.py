import argparse
from collections import namedtuple
import csv
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
    'num_models',
    'trial_idx',
    'workspace',
    'eval_correct',
    'train_proportion',
    'root_outdir',
])


class Classifier:
    def __init__(self, num_models):
        self.clfs = [MLPClassifier() for _ in range(num_models)]

    def fit(self, X, y):
        for idx, clf in enumerate(self.clfs):
            start = idx * REPRESENTATIONS_SIZE
            end = start + REPRESENTATIONS_SIZE
            clf.fit(X[:, start:end], y)

    def predict_proba(self, X):
        y_probas = []
        for idx, clf in enumerate(self.clfs):
            start = idx * REPRESENTATIONS_SIZE
            end = start + REPRESENTATIONS_SIZE
            y_proba = clf.predict_proba(X[:, start:end])
            y_probas.append(y_proba)
        return np.mean(y_probas, axis=0)


def run(params):
    with LOCK:
        print('run_idx:', run_idx.value, 'num_models:', params.num_models, 'trial_idx:', params.trial_idx)
        run_idx.value += 1
    set_seed(params.seed)
    with LOCK:
        # Draw a random sample of non-attacked models.
        models = sorted(list(np.random.choice(range(1, NUM_MODELS + 1), size=params.num_models, replace=False)))
        representations = []
        adv_repr_lookup = {attack: [] for attack in ATTACKS}
        # WARN: The data in representations.hdf5 is transposed for faster loading.
        for model in models:
            repr_path = os.path.join(params.workspace, 'eval', 'by_model', str(model), 'representations.hdf5')
            with h5py.File(repr_path, 'r') as f:
                repr_subset = f['representations']
                assert repr_subset.shape[0] == REPRESENTATIONS_SIZE
                repr_subset = repr_subset[:].T
                # Limit to images that were correctly classified initially.
                # This was already done earlier in the pipeline for the adversarial images.
                # This is not included as part of the indexing above, as it's faster this way.
                repr_subset = repr_subset[np.where(params.eval_correct)[0]]
                assert type(repr_subset) == np.ndarray
            representations.append(repr_subset)
            for attack in ATTACKS:
                adv_repr_path = os.path.join(
                    params.workspace, 'attack', attack, 'by_model', str(model), 'representations.hdf5')
                with h5py.File(adv_repr_path, 'r') as f:
                    adv_repr_subset = f['representations']
                    assert adv_repr_subset.shape[0] == REPRESENTATIONS_SIZE
                    adv_repr_subset = adv_repr_subset[:].T
                    assert type(adv_repr_subset) == np.ndarray
                    assert repr_subset.shape == adv_repr_subset.shape
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
        clf = Classifier(params.num_models)
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
            csv_path = os.path.join(outdir, f'{params.num_models}.csv')
            with LOCK, open(csv_path, 'a') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    ','.join(str(x) for x in models),  # models
                    sum(correct),                      # correct
                    accuracy                           # accuracy
                ])


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-trials', type=int, default=100)
    parser.add_argument('--max-num-models', type=int, default=10)
    parser.add_argument('--train-proportion', type=float, default=0.90)
    parser.add_argument('--num-jobs', type=int, default=multiprocessing.cpu_count())
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    attacked_model = 0
    eval_correct_path = os.path.join(
        args.workspace, 'eval', 'by_model', str(attacked_model), 'correct.csv')
    eval_correct = np.loadtxt(eval_correct_path, dtype=bool, delimiter=',')
    root_outdir = os.path.join(args.workspace, 'detect_model_wise_treatment')
    os.makedirs(root_outdir, exist_ok=True)
    num_models_range = range(1, args.max_num_models + 1)
    for adv_source in ATTACKS:
        for adv_target in ATTACKS:
            outdir = os.path.join(root_outdir, adv_source, adv_target)
            os.makedirs(outdir, exist_ok=True)
            for num_models in num_models_range:
                with open(os.path.join(outdir, f'{num_models}.csv'), 'w') as f:
                    f.write('models,correct,accuracy\n')
    params_list = []
    for num_models in num_models_range:
        for trial_idx in range(args.num_trials):
            params = Params(
                seed=args.seed + len(params_list),
                num_models=num_models,
                trial_idx=trial_idx,
                workspace=args.workspace,
                eval_correct=eval_correct,
                train_proportion=args.train_proportion,
                root_outdir=root_outdir
            )
            params_list.append(params)
    # Shuffle parameters list so that run_idx/num_runs more closely represents percentage
    # complete 'time'.
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
        'num_models': [],
        'accuracy_mean': [],
        'accuracy_std': [],
        'accuracy_count': []
    }
    for adv_source in ATTACKS:
        for adv_target in ATTACKS:
            for num_models in num_models_range:
                df = pd.read_csv(os.path.join(root_outdir, adv_source, adv_target, f'{num_models}.csv'))
                aggregated_dict['adv_source'].append(adv_source)
                aggregated_dict['adv_target'].append(adv_target)
                aggregated_dict['num_models'].append(num_models)
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
