import os
from functools import partial

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, distributed
from tinyimagenet import TrainTinyImageNetDataset, TestTinyImageNetDataset, CorruptTinyImageNetDataset

VALID_SPLIT_SEED=88
NORM_STAT = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'vgg_cifar10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'vgg_cifar100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'svhn': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'tinyimagenet': ((122.4786, 114.2755, 101.3963), (70.4924, 68.5679, 71.8127))
}
NUM_CLASSES = {
    'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200
}
def get_data_loader(dataset, norm_stat=None, train_bs=64, test_bs=64, validation=False, validation_fraction=0.1, root_dir='data/', test_only=False, train_only=False, augment=True,
                    num_train_workers=2, num_test_workers=2, shuffle_train=True, drop_last_train=True):
    if dataset in ('cifar10', 'cifar100', 'svhn'):
        data_cls = getattr(torchvision.datasets, dataset.upper())
    if dataset in ('vgg_cifar10', 'vgg_cifar100'):
        data_cls = getattr(torchvision.datasets, dataset[4:].upper())
    if dataset in ('cifar10', 'cifar100', 'vgg_cifar10', 'vgg_cifar100') :
        train_data_cls = partial(data_cls, train=True, root=root_dir, download=True)
        test_data_cls = partial(data_cls, train=False, root=root_dir, download=True)
        augment_transform = [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ] if augment else []
    if dataset == 'svhn':
        train_data_cls = partial(data_cls, split='train', root=root_dir, download=True)
        test_data_cls = partial(data_cls, split='test', root=root_dir, download=True)
    if dataset == 'tinyimagenet':
        train_data_cls = TrainTinyImageNetDataset
        test_data_cls = TestTinyImageNetDataset
        augment_transform = [
            torchvision.transforms.RandomCrop(64, padding=4),
            torchvision.transforms.RandomHorizontalFlip()
        ] if augment else []
    transform = torchvision.transforms.Compose([
        *([torchvision.transforms.ToTensor()] if dataset in ('cifar10', 'cifar100', 'svhn', 'vgg_cifar10', 'vgg_cifar100') else []),
        torchvision.transforms.Normalize(*(NORM_STAT[dataset] if norm_stat is None else norm_stat))
    ])
    train_data = train_data_cls(
        transform=torchvision.transforms.Compose([
            *augment_transform,
            transform
        ]))
    if train_only:
        train_loader = DataLoader(train_data, batch_size=train_bs, pin_memory=True, shuffle=shuffle_train, drop_last=drop_last_train, num_workers=num_train_workers)
        return train_loader
    test_data = test_data_cls(transform=transform)
    test_loader = DataLoader(test_data, batch_size=test_bs, pin_memory=True, shuffle=False, num_workers=num_test_workers)
    if test_only:
        return test_loader
    if validation:
        valid_data = train_data_cls(transform=transform)
        train_idx, valid_idx = train_test_split(np.arange(len(train_data.targets)),
                                                test_size=validation_fraction,
                                                shuffle=True, random_state=VALID_SPLIT_SEED,
                                                stratify=train_data.targets)
        train_loader = DataLoader(Subset(train_data, train_idx), batch_size=train_bs, pin_memory=True, shuffle=shuffle_train, drop_last=drop_last_train, num_workers=num_train_workers)
        valid_loader = DataLoader(Subset(valid_data, valid_idx), batch_size=test_bs, pin_memory=True, shuffle=False, drop_last=False, num_workers=num_test_workers)
        return train_loader, valid_loader, test_loader
    else:
        train_loader = DataLoader(train_data, batch_size=train_bs, pin_memory=True, shuffle=shuffle_train, drop_last=drop_last_train, num_workers=num_train_workers)
        return train_loader, test_loader


class CorruptDataset(torch.utils.data.Dataset):
    def __init__(self, root, corrupt_types, intensity, transform=None):
        self.data = np.concatenate(
            [np.load(os.path.join(root, f'{corrupt_type}.npy'))[intensity*10000:(intensity+1)*10000] for corrupt_type in corrupt_types], axis=0
        )
        self.label = np.concatenate(
            [np.load(os.path.join(root, 'labels.npy'))[intensity*10000:(intensity+1)*10000]] * len(corrupt_types), axis=0
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx], int(self.label[idx])
        sample = Image.fromarray(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label
 
def get_corrupt_data_loader(dataset, intensity, batch_size=64, root_dir='data/', num_workers=4, norm_stat=None):
    corrupt_type = ['saturate',
                    'shot_noise',
                    'gaussian_noise',
                    'zoom_blur',
                    'glass_blur',
                    'brightness',
                    'contrast',
                    'motion_blur',
                    'pixelate',
                    'snow',
                    'speckle_noise',
                    'spatter',
                    'gaussian_blur',
                    'frost',
                    'defocus_blur',
                    'elastic_transform',
                    'impulse_noise',
                    'jpeg_compression',
                    'fog']
    
    transform = torchvision.transforms.Compose([
        *([torchvision.transforms.ToTensor()] if dataset in ('cifar10', 'cifar100') else []),
         torchvision.transforms.Normalize(*(NORM_STAT[dataset] if norm_stat is None else norm_stat))
    ])
    if dataset == 'cifar10':
        test_data = CorruptDataset(os.path.join(root_dir, 'CIFAR-10-C'), corrupt_type, intensity, transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)
        return test_loader
    if dataset == 'cifar100':
        test_data = CorruptDataset(os.path.join(root_dir, 'CIFAR-100-C'), corrupt_type, intensity, transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)
        return test_loader
    if dataset == 'tinyimagenet':
        test_data = CorruptTinyImageNetDataset(intensity+1, transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers)
        return test_loader