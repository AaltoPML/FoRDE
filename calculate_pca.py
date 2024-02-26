import torchvision
import numpy as np
from datasets import NORM_STAT
import argparse
from tinyimagenet import TrainTinyImageNetDataset
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'])
    parser.add_argument('save_path', type=str)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    if args.dataset == 'cifar10':
        data = torchvision.datasets.CIFAR10(root='data', train=True, download=True).data / 255.0
    elif args.dataset == 'cifar100':
        data = torchvision.datasets.CIFAR100(root='data', train=True, download=True).data / 255.0
    elif args.dataset == 'tinyimagenet':
        data = []
        dataset = TrainTinyImageNetDataset()
        for image, _ in dataset:
            data.append(
                torch.permute(image, (1, 2, 0)).numpy()
            )
        data = np.stack(data, axis=0)
    
    data = (data-NORM_STAT[args.dataset][0])/NORM_STAT[args.dataset][1]
    data = np.reshape(data, (data.shape[0], -1))
    empcov = (data.T @ data)/(data.shape[0]-1)
    U, S, Vh = np.linalg.svd(empcov)
    np.savez(args.save_path, eigenvectors=U.astype(np.float32), eigenvalues=np.sqrt(S).astype(np.float32))