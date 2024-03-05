---
title: Input-gradient space particle inference for neural network ensembles
publication: Spotlight (top 5% papers) at the Twelfth International Conference on Learning Representations (ICLR) 2024
description: Trung Trinh, Markus Heinonen, Luigi Acerbi, Samuel Kaski
---

*This website contains information regarding the paper Input-gradient space particle inference for neural network ensembles.*

> **TL;DR**: We introduce First-order Repulsive Deep ensembles (FoRDEs), which are ensembles of neural networks with diversified input gradients.

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
# Particle-based variational inference for neural network ensembles

Ensemble methods, which combine predictions from multiple models, are a well-known strategy in machine learning to boost predictive performance, uncertainty estimation and robustness under covariate shifts.
The successes of ensemble methods are mainly due to the _functional diversity of their members_.
For neural networks, one can create an ensemble by training multiple neural networks from different independent random initializations, a strategy called Deep ensembles (DEs).
The effectiveness of a DE depends on the randomness of the training procedure to implicitly induce weight-space diversity, as independent training runs under different random conditions will likely converge to different modes in the weight posterior distribution.
However, weight diversities do not necessarily translate into useful functional diversities due to the inherent symmetries in the weight space, i.e., two sets of weights can represent the same function.




# First-order Repulsive deep ensembles

## Wasserstein gradient descent (WGD)