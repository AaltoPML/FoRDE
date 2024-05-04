---
title: Input-gradient space particle inference for neural network ensembles
publication: Spotlight (top 5% papers) at the Twelfth International Conference on Learning Representations (ICLR) 2024
description: Trung Trinh, Markus Heinonen, Luigi Acerbi, Samuel Kaski
---

*This website contains information regarding the paper Input-gradient space particle inference for neural network ensembles.*

> **TL;DR**: We introduce First-order Repulsive Deep ensembles (FoRDEs), a method that trains an ensemble of neural networks diverse with respect to their input gradients.

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

# Repulsive deep ensembles (RDEs) [1]

> **Description:** Train an ensemble \\(\\{\boldsymbol{\theta}\_i\\}_{i=1}^M\\) using Wasserstein gradient descent, which employs a <span class="my_blue">kernelized repulsion term</span> to diversify the particles to cover the <span class="my_red"> Bayes posterior \\(p(\boldsymbol{\theta} \| \mathcal{D}) \\)</span>. 

\begin{equation}
\boldsymbol{\theta}\_i^{(t+1)} = \boldsymbol{\theta}\_i^{(t)} + \eta\_t\bigg( 
      {\color{red}
\underbrace{
\nabla\_{\boldsymbol{\theta}\_i^{(t)}} \log p(\boldsymbol{\theta}\_i^{(t)} \| \mathcal{D}) 
}\_{\text{Driving force}}}
      -
      {\color[RGB]{68,114,196}
        \underbrace{\frac{
          \sum\_{j=1}^N \nabla\_{\boldsymbol{\theta}\_i^{(t)}} k(\boldsymbol{\theta}\_i^{(t)}, \boldsymbol{\theta}\_j^{(t)})
        }{
           \sum\_{j=1}^N k(\boldsymbol{\theta}\_i^{(t)}, \boldsymbol{\theta}\_j^{(t)})
        }}\_{\text{Repulsion force}}
      }
    \bigg)
\end{equation}