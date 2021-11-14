import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import torch

from model import Net
from utils import cifar10_loader, DEFAULT_WORKSPACE, get_devices, NUM_MODELS


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workspace', type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument('--num-saved-images', type=int, help='Max number of images to save per class.')
    devices = get_devices()
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    eval_dir = os.path.join(args.workspace, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    model_range = range(NUM_MODELS + 1)
    attacked_idx = 0
    for model_idx in model_range:
        print(f'idx: {model_idx}')
        outdir = os.path.join(eval_dir, 'by_model', str(model_idx))
        os.makedirs(outdir, exist_ok=True)
        net = Net().to(args.device)
        net_path = os.path.join(args.workspace, 'networks', f'{model_idx}.pt')
        net.load_state_dict(torch.load(net_path, map_location='cpu'))
        net.eval()
        test_loader = cifar10_loader(args.batch_size, train=False, shuffle=False)
        classes = test_loader.dataset.classes
        saved_img_counts = [0] * 10
        y = []
        y_pred = []
        y_pred_proba = []
        y_repr = []
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            y.extend(y_batch.tolist())
            outputs, representations = net(x_batch, include_penultimate=True)
            outputs = outputs.detach().cpu().numpy()
            representations = representations.detach().cpu().numpy()
            y_pred.extend(outputs.argmax(axis=1))
            y_pred_proba.extend(softmax(outputs, axis=1).tolist())
            y_repr.extend(representations.tolist())
            # Save example images.
            if model_idx == 0:
                for image_idx, class_ in enumerate(y_batch.tolist()):
                    if args.num_saved_images is not None and saved_img_counts[class_] >= args.num_saved_images:
                        continue
                    img_dir = os.path.join(eval_dir, 'images', f'{class_}_{classes[class_]}')
                    os.makedirs(img_dir, exist_ok=True)
                    img_arr = (x_batch[image_idx].detach().cpu().numpy() * 255).round().astype(np.uint8).transpose([1, 2, 0])
                    img = Image.fromarray(img_arr)
                    img_id = test_loader.batch_size * batch_idx + image_idx
                    img.save(os.path.join(img_dir, f'{img_id}.png'))
                    saved_img_counts[class_] += 1
        y = np.array(y)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        correct = y_pred == y
        y_repr = np.array(y_repr)
        np.savetxt(os.path.join(outdir, 'ground_truth.csv'), y, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(outdir, 'pred.csv'), y_pred, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(outdir, 'pred_proba.csv'), y_pred_proba, delimiter=',', fmt='%f')
        np.savetxt(os.path.join(outdir, 'correct.csv'), correct, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(outdir, 'representations.csv'), y_repr, delimiter=',', fmt='%f')
        # Save transposed representations as HDF5 for quicker parsing/loading later.
        with h5py.File(os.path.join(outdir, 'representations.hdf5'), 'w') as f:
            f.create_dataset('representations', data=y_repr.T)
        cm = confusion_matrix(y, y_pred)
        print('Confusion Matrix:')
        print(cm)
        np.savetxt(os.path.join(outdir, 'confusion.csv'), cm, delimiter=',', fmt='%d')
        num_correct = correct.sum()
        total = len(y_pred)
        accuracy = num_correct / total
        eval_dict = {
            'correct': [num_correct],
            'total': [total],
            'accuracy': [accuracy]
        }
        eval_df = pd.DataFrame.from_dict(eval_dict)
        print('Evaluation:')
        print(eval_df)
        eval_df.to_csv(os.path.join(outdir, 'eval.csv'), index=False)
    # Aggregate
    print('Aggregate')
    accuracies = []
    for model_idx in model_range:
        # Don't include the attacked model when aggregating accuracies.
        if model_idx == attacked_idx:
            continue
        by_model_dir = os.path.join(eval_dir, 'by_model', str(model_idx))
        eval_df = pd.read_csv(os.path.join(by_model_dir, 'eval.csv'))
        accuracies.append(eval_df.accuracy.item())
    aggregated_dict = {
        'accuracy_mean': [np.mean(accuracies)],
        'accuracy_std': [np.std(accuracies, ddof=1)],
        'accuracy_count': [len(accuracies)]
    }
    aggregated_df = pd.DataFrame.from_dict(aggregated_dict)
    print(aggregated_df)
    aggregated_path = os.path.join(eval_dir, 'aggregated.csv')
    aggregated_df.to_csv(aggregated_path, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
