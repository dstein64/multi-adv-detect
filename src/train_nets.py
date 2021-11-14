import argparse
import json
import os
import sys

import torch

from model import Net
from utils import cifar10_classes, cifar10_loader, DEFAULT_WORKSPACE, get_devices, NUM_MODELS, set_seed


def main(argv=sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--end-seed', type=int, default=NUM_MODELS)
    parser.add_argument('--workspace', type=str, default=DEFAULT_WORKSPACE)
    devices = get_devices()
    parser.add_argument('--device', default='cuda' if 'cuda' in devices else 'cpu', choices=devices)
    args = parser.parse_args(argv[1:])
    os.makedirs(args.workspace, exist_ok=True)
    with open(os.path.join(args.workspace, 'classes.json'), 'w') as f:
        json.dump(cifar10_classes(), f, indent=2)
    nets_path = os.path.join(args.workspace, 'networks')
    os.makedirs(nets_path, exist_ok=True)
    for idx, seed in enumerate(range(args.start_seed, args.end_seed + 1)):
        print(f'idx: {idx}, seed: {seed}')
        set_seed(seed)
        net = Net().to(args.device)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(net.parameters())
        net.train()
        train_loader = cifar10_loader(args.batch_size, train=True)
        for epoch in range(1, args.epochs + 1):
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                loss = loss_fn(net(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'epoch: {epoch}/{args.epochs}, train loss: {train_loss:.3f}')
        net_path = os.path.join(nets_path, f'{seed}.pt')
        torch.save(net.state_dict(), net_path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
