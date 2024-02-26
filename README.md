# Input-gradient space particle inference for neural network ensembles

This repository contains a Jax/Flax implementation of the paper

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

## Installation

Please follow the instructions in `install_steps.txt` to install the necessary python packages.

## Command to replicate the result

For more information on each training option, please read the comments in the `train_forde.py` file.
Each experiment will be stored in a subfolder of the `experiments` folder.
