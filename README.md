# Input-gradient space particle inference for neural network ensembles

This repository contains a Jax/Haiku implementation of the paper

[Input-gradient space particle inference for neural network ensembles](https://openreview.net/forum?id=nLWiR5P3wr)

by Trung Trinh, Markus Heinonen, Luigi Acerbi and Samuel Kaski

For more information about the paper, please visit the [website](https://aaltopml.github.io/FoRDE/).

Please cite our work if you find it useful:

```bibtex
@inproceedings{trinh2024inputgradient,
    title={Input-gradient space particle inference for neural network ensembles},
    author={Trung Trinh and Markus Heinonen and Luigi Acerbi and Samuel Kaski},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=nLWiR5P3wr}
}
```

## Setting up

### Installing the python packages
Please follow the instructions in `install_steps.txt` to install the necessary python packages.

### Downloading the datasets
To run the experiments, one needs to run the following commands to download the necessary datasets and store them in the `data` folder:
```bash
bash download_scripts/download_cifar10_c.sh
bash download_scripts/download_cifar100_c.sh
bash download_scripts/download_tinyimagenet.sh
bash download_scripts/download_tinyimagenet_c.sh
```

## Instructions to replicate the results
To replicate the results, we need two steps:

1. Calculate the PCA of the training data using the `calculate_pca.py` script.
2. Train a FoRDE using the `train_forde.py` script.

For instance, to replicate the results of `ResNet18/CIFAR100` experiments, we first run:
```bash
mkdir data_pca
python calculate_pca.py cifar100 data_pca/cifar100.npz
```
where the first argument is the dataset and the second argument is the path to save the PCA of that dataset. Then, we run the following command to train a FoRDE:
```bash
python train_forde.py with model_name=ResNet18 seed=${SEED} validation=False \
                    "name=e300_10_eps1e-24_forde_wd5e-4_lrratio0.01" \
                    batch_size=128 num_start_epochs=0 num_train_workers=4 num_test_workers=4 \
                    "data_pca_path=data_pca/cifar100.npz" n_members=10 init_lr=0.10 lr_ratio=0.01 \
                    dataset=cifar100 num_epochs=300 "weight_decay=5e-4" "eps=1e-24"
```
where `${SEED}` is a chosen random seed for the run.

Similarly, to replicate the `ResNet18/CIFAR10` experiments, we run the following commands:
```bash
mkdir data_pca
python calculate_pca.py cifar10 data_pca/cifar10.npz
python train_forde.py with model_name=ResNet18 seed=${SEED} validation=False \
                    "name=e300_10_eps1e-24_forde_wd5e-4_lrratio0.01" \
                    batch_size=128 num_start_epochs=0 num_train_workers=4 num_test_workers=4 \
                    "data_pca_path=data_pca/cifar10.npz" n_members=10 init_lr=0.10 lr_ratio=0.01 \
                    dataset=cifar10 num_epochs=300 "weight_decay=5e-4" "eps=1e-24"
```
and to replicate the `PreActResNet18/TinyImageNet` experiments, we run the following commands:
```bash
mkdir data_pca
python calculate_pca.py tinyimagenet data_pca/tinyimagenet.npz
python train_forde.py with model_name=PreActResNet18 seed=${SEED} validation=False \
                    "name=e150_10_eps1e-24_forde_wd5e-4_lrratio0.001" \
                    batch_size=128 num_start_epochs=0 num_train_workers=4 num_test_workers=4 \
                    "data_pca_path=data_pca/tinyimagenet.npz" n_members=10 init_lr=0.10 lr_ratio=0.001 \
                    dataset=tinyimagenet num_epochs=150 "weight_decay=5e-4" "eps=1e-24"

```

For more information on each training option, please read the comments in the `train_forde.py` file.
Each experiment will be stored in a subfolder of the `experiments` folder.
