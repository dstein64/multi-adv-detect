import argparse
import os
import sys
import time

from cleverhans.torch.attacks import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import torch

from model import Net
from utils import ATTACKS, cifar10_loader, DEFAULT_WORKSPACE, get_devices, NUM_MODELS, set_seed


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workspace', type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        '--num-saved-images', type=int, help='Max number of images to save per class for each attack.')
    devices = get_devices()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    attack_dir = os.path.join(args.workspace, 'attack')
    os.makedirs(attack_dir, exist_ok=True)
    set_seed(args.seed)
    attacked_idx = 0
    attacked_net = Net().to(args.device)
    attacked_net_path = os.path.join(args.workspace, 'networks', f'{attacked_idx}.pt')
    attacked_net.load_state_dict(torch.load(attacked_net_path, map_location='cpu'))
    attacked_net.eval()
    test_loader = cifar10_loader(args.batch_size, train=False, shuffle=False)
    classes = test_loader.dataset.classes
    attacked_eval_correct_path = os.path.join(
        args.workspace, 'eval', 'by_model', str(attacked_idx), 'correct.csv')
    attacked_eval_correct = np.loadtxt(attacked_eval_correct_path, dtype=bool, delimiter=',')
    model_range = range(NUM_MODELS + 1)
    for attack in ATTACKS:
        print('attack:', attack, time.time())
        outdir = os.path.join(args.workspace, 'attack', attack)
        os.makedirs(outdir, exist_ok=True)
        saved_img_ids = list(reversed(np.where(attacked_eval_correct)[0]))
        saved_img_counts = [0] * 10
        norms = {'l0': [], 'l1': [], 'l2': [], 'linf': []}
        y = []
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            print('  batch_idx:', batch_idx, time.time())
            # Limit attack to images that were correctly classified initially by the attacked model.
            offset = test_loader.batch_size * batch_idx
            correct_idxs = np.where(attacked_eval_correct[offset:offset + len(x_batch)])[0]
            if correct_idxs.size == 0:
                continue
            x_batch, y_batch = x_batch[correct_idxs], y_batch[correct_idxs]
            x_batch, y_batch = x_batch.to(args.device), y_batch.to(args.device)
            if attack == 'fgsm':
                # Modify each pixel by up to 3 intensity values.
                x_adv_batch = fast_gradient_method(attacked_net, x_batch, 3 / 255, float('inf'))
            elif attack == 'bim':
                # Modify each pixel by up to 1 intensity value per iteration, for 10 iterations.
                # Clip to 3 intensity values.
                x_adv_batch = projected_gradient_descent(
                    attacked_net, x_batch, 3 / 255, 1 / 255, 10, float('inf'), rand_init=False)
            elif attack == 'cw':
                # The default confidence is 0. Increasing confidence results in a larger perturbation
                # that is more transferable (see Section VI and VIII-D of C&W).
                x_adv_batch = carlini_wagner_l2(attacked_net, x_batch, 10, confidence=100)
            else:
                raise RuntimeError('Unsupported attack: ' + attack)
            # Match the quantization of the non-adversarial images. For C&W with a low or zero setting
            # for 'confidence', quantizing could remove the effectiveness of the attack. This is not
            # an issue for the confidence setting used above.
            x_adv_batch = ((x_adv_batch * 255).round() / 255.0)
            x_adv_batch = x_adv_batch.clip(0, 1)
            perturb_batch = (x_batch - x_adv_batch).flatten(start_dim=1)
            for p in [0, 1, 2, float('inf')]:
                norms[f'l{p}'].extend(perturb_batch.norm(p=p, dim=1).tolist())
            y.extend(y_batch.detach().cpu())
            # Pass batch through each network, saving outputs and representations.
            # (each loop iteration takes about 0.6 seconds for batches of 1000 images)
            print('    pass through networks', time.time())
            for net_seed in model_range:
                net = Net().to(args.device)
                net_path = os.path.join(args.workspace, 'networks', f'{net_seed}.pt')
                net.load_state_dict(torch.load(net_path, map_location='cpu'))
                net.eval()
                outputs_batch, representations_batch = net(x_adv_batch, include_penultimate=True)
                outputs_batch = outputs_batch.detach().cpu().numpy()
                representations_batch = representations_batch.detach().cpu().numpy()
                y_pred_batch = outputs_batch.argmax(axis=1)
                y_pred_proba_batch = softmax(outputs_batch, axis=1)
                mode = 'w' if batch_idx == 0 else 'a'
                net_outdir = os.path.join(outdir, 'by_model', str(net_seed))
                os.makedirs(net_outdir, exist_ok=True)
                with open(os.path.join(net_outdir, 'pred.csv'), mode) as f:
                    np.savetxt(f, y_pred_batch, delimiter=',', fmt='%d')
                with open(os.path.join(net_outdir, 'pred_proba.csv'), mode) as f:
                    np.savetxt(f, y_pred_proba_batch, delimiter=',', fmt='%f')
                with open(os.path.join(net_outdir, 'representations.csv'), mode) as f:
                    np.savetxt(f, representations_batch, delimiter=',', fmt='%f')
            # Save example perturbed images.
            for idx, class_ in enumerate(y_batch.tolist()):
                if args.num_saved_images is not None and saved_img_counts[class_] >= args.num_saved_images:
                    continue
                if idx == 0:
                    print('    saving images', time.time())
                img_dir = os.path.join(outdir, 'images', f'{class_}_{classes[class_]}')
                os.makedirs(img_dir, exist_ok=True)
                img_arr = (x_adv_batch[idx].detach().cpu().numpy() * 255).round().astype(np.uint8).transpose([1, 2, 0])
                img = Image.fromarray(img_arr)
                img.save(os.path.join(img_dir, f'{saved_img_ids.pop()}.png'))
                saved_img_counts[class_] += 1
        y = np.array(y)
        np.savetxt(os.path.join(outdir, 'ground_truth.csv'), y, delimiter=',', fmt='%d')
        norms_df = pd.DataFrame.from_dict(norms)
        norms_df.to_csv(os.path.join(outdir, 'norms.csv'), index=False)
        norms_df.describe().to_csv(os.path.join(outdir, 'norms_stats.csv'), index=False)
        # Generate evaluations for each network, including the attacked model.
        for net_seed in model_range:
            net_outdir = os.path.join(outdir, 'by_model', str(net_seed))
            y_pred = np.loadtxt(os.path.join(net_outdir, 'pred.csv'), dtype=int)
            correct = y_pred == y
            np.savetxt(os.path.join(net_outdir, 'correct.csv'), correct, delimiter=',', fmt='%d')
            cm = confusion_matrix(y, y_pred)
            np.savetxt(os.path.join(net_outdir, 'confusion.csv'), cm, delimiter=',', fmt='%d')
            num_correct = correct.sum()
            total = len(y_pred)
            accuracy = num_correct / total
            eval_dict = {
                'correct': [num_correct],
                'total': [total],
                'accuracy': [accuracy]
            }
            eval_df = pd.DataFrame.from_dict(eval_dict)
            eval_df.to_csv(os.path.join(net_outdir, 'eval.csv'), index=False)
        # Save transposed representations as HDF5 for quicker parsing/loading later.
        # (each loop iteration takes about 1.6 seconds)
        print('  save representations.hdf5', time.time())
        for net_seed in model_range:
            net_outdir = os.path.join(outdir, 'by_model', str(net_seed))
            representations = np.loadtxt(os.path.join(net_outdir, 'representations.csv'), delimiter=',')
            with h5py.File(os.path.join(net_outdir, 'representations.hdf5'), 'w') as f:
                f.create_dataset('representations', data=representations.T)
    # Aggregate
    print('Aggregate')
    accuracies = {attack: {} for attack in ATTACKS}
    for attack in ATTACKS:
        for model_idx in model_range:
            by_model_dir = os.path.join(
                args.workspace, 'attack', attack, 'by_model', str(model_idx))
            eval_df = pd.read_csv(os.path.join(by_model_dir, 'eval.csv'))
            accuracies[attack][model_idx] = eval_df.accuracy.item()
    aggregated_dict = {
        'attack': [],
        'accuracy_mean': [],
        'accuracy_std': [],
        'accuracy_count': []
    }
    for attack in ATTACKS:
        aggregated_dict['attack'].append(attack)
        # Don't include the attacked model when aggregating accuracies.
        skip_target_acc = [acc for seed, acc in accuracies[attack].items() if seed != attacked_idx]
        aggregated_dict['accuracy_mean'].append(np.mean(skip_target_acc))
        aggregated_dict['accuracy_std'].append(np.std(skip_target_acc, ddof=1))
        aggregated_dict['accuracy_count'].append(len(skip_target_acc))
    aggregated_df = pd.DataFrame.from_dict(aggregated_dict)
    print(aggregated_df)
    aggregated_path = os.path.join(attack_dir, 'aggregated.csv')
    aggregated_df.to_csv(aggregated_path, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
