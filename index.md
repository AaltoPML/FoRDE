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
The successes of ensemble methods are mainly due to the **functional diversity of their members**.
For neural networks, one can create an ensemble by training multiple neural networks from different independent random initializations, a strategy called Deep ensembles (DEs).
The effectiveness of a DE depends on the randomness of the training procedure to implicitly induce weight-space diversity, as independent training runs under different random conditions will likely converge to different modes in the weight posterior distribution.
However, weight diversities do not necessarily translate into useful functional diversities due to the inherent symmetries in the weight space, i.e., two sets of weights can represent the same function.

To explicitly promote diversity in a neural network ensemble, **particle-based variational inference (ParVI)** has recently emerged as a promising approach.
Notably, the ParVI update rule includes a kernelized repulsion term \\(k(f, f^\prime)\\) between ensemble members \\(f, f^\prime\\) to control the diversity.
Current approaches compare networks in weight space or function space.
Weight-space repulsion is ineffective due to the extremely high dimensionality and symmetries of the weight posterior.
Comparing neural networks via a function kernel is also challenging since functions are infinite-dimensional objects. Previous works resort to comparing functions only on a subset of the input space. Comparing functions
over training data leads to underfitting, likely because these inputs have known labels, leaving no room for diverse predictions without impairing performance.
Neither weight nor function space repulsion has led to significant improvements over vanilla DEs.

# FoRDEs: First-order Repulsive deep ensembles 

From a functional perspective, a model can also be uniquely represented, up to translation, using
its first-order derivatives, i.e., input gradients \\(\nabla_{\mathbf{x}} f\\). Promoting diversity in this third view of input gradients has notable advantages:

<ol>
  <li>each ensemble member is guaranteed to correspond to a different function;</li>
  <li>input gradients have smaller dimensions than weights and thus are more amenable to kernel
comparisons;</li>
  <li>unlike function-space repulsion, input-gradient repulsion does not lead to training point
underfitting</li>
  <li>each ensemble member is encouraged to learn different features, which can improve robustness.</li>
</ol> 

Thus, we propose First-order Repulsive deep ensembles (FoRDEs), which are ParVI neural network ensembles that promote diversity in their input gradients. In the following sections, we present the training algorithm, the formulation of the kernel for the repulsion term as well as how to select the hyperparameters.

Below, we assume a set of \\(M\\) weight particles \\(\{\theta_i\}_{i=1}^M\\) corresponding to a set of \\(M\\) neural networks \\(\{f_i: \mathbf{x} \mapsto f(\mathbf{x}; \theta_i)\}_{i=1}^M\\).
We focus on the supervised classification setting: given a labelled dataset \\(\mathcal{D}=\{(\mathbf{x}_n, y_n)\}_{n=1}^N\\) with \\(\mathcal{C}\\) classes and inputs \\(\mathbf{x}_n \in \R^D\\), we approximate the posterior \\(p(\theta | \mathcal{D})\\) using the \\(M\\) particles. The output \\(f(\mathbf{x}; \theta)\\) for input \\(\mathbf{x}\\) is a vector of size \\(\mathcal{C}\\) whose \\(y\\)-th entry \\(f(\mathbf{x}; \theta)_y\\) is the logit of the \\(y\\)-th class.

## Training algorithm: Wasserstein gradient descent (WGD)
We use a ParVI method called Wasserstein gradient descent (WGD), which updates the weight particles \\(\{\theta_i\}_{i=1}^M\\) using the following rule:

$$\theta^{(t+1)} = \theta^{(t)} + \eta_t \Bigg(\underbrace{\nabla_{\theta^{(t)}} \log p\big(\theta^{(t)}|\mathcal{D}\big)}_{\text{driving force}} - \underbrace{\frac{\sum_{i=1}^M \nabla_{\theta^{(t)}} k\big(\theta^{(t)},\theta_i^{(t)}\big)}{\sum_{i=1}^M k\big(\theta^{(t)},\theta_i^{(t)}\big)}}_{\text{repulsion force}} \Bigg)$$

where \\(\eta_t >0\\) is the step size at optimization step \\(t\\). Intuitively, the first term can be interpreted as the driving force directing the particles towards high density regions of the posterior \\(p(\theta|\mathcal{D})\\), while the second term is the repulsion force pushing the particles away from each other.

## Formulating the kernel for input-gradient space repulsion
We propose to use a kernel comparing the **input gradients** of the particles,

$$k(\theta_i, \theta_j) \overset{\mathrm{def}}{=} \mathbb{E}_{(\mathbf{x},y) \sim p(\mathbf{x},y)}\Big[ \kappa\big(\nabla_\mathbf{x} f(\mathbf{x}; \theta_i)_{y}, \nabla_\mathbf{x} f(\mathbf{x}; \theta_j)_{y} \big) \Big],$$

where \\(\kappa\\) is a **base kernel** between gradients \\(\nabla_\mathbf{x} f(\mathbf{x};\theta)_y\\) that are of same size as the inputs \\(\mathbf{x}\\). 
During training, we approximate the kernel \\(k\\) using the training samples, with linear complexity:

$$k(\theta_i, \theta_j) \approx k_{\mathcal{D}}(\theta_i, \theta_j) = \frac{1}{N}\sum_{n=1}^N \kappa\big(\nabla_\mathbf{x} f(\mathbf{x}_n; \theta_i)_{y_n}, \nabla_\mathbf{x} f(\mathbf{x}_n; \theta_j)_{y_n} \big).$$

The kernel only compares the gradients of the true label \\(\nabla_\mathbf{x} f(\mathbf{x}_n; \theta)_{y_n}\\), as opposed to the entire Jacobian matrix \\(\nabla_\mathbf{x} f(\mathbf{x}_n; \theta)\\), as our motivation is to encourage each particle to learn different features that could explain the training sample \\((\mathbf{x}_n,y_n)\\) well.
This approach also reduces computational complexity, since automatic differentiation libraries such as JAX or Pytorch would require \\(\mathcal{C}\\) passes, one per class, to calculate the full Jacobian.