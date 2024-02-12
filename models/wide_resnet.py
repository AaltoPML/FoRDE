# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WideResNet implementation in JAX using Haiku."""

from typing import Any, Dict, Optional

import haiku as hk
import jax
import jax.numpy as jnp


class _WideResNetBlock(hk.Module):
  """Block of a WideResNet."""

  def __init__(self, num_filters, stride=1, projection_shortcut=False,
               activation=jax.nn.relu, bn_config=None, name=None):
    super().__init__(name=name)
    num_bottleneck_layers = 1
    self._activation = activation
    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.99)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", False)
    self._bn_modules = []
    self._conv_modules = []
    for i in range(num_bottleneck_layers + 1):
      s = stride if i == 0 else 1
      self._bn_modules.append(hk.BatchNorm(
          name='batchnorm_{}'.format(i),
          **bn_config))
      self._conv_modules.append(hk.Conv2D(
          output_channels=num_filters,
          padding='SAME',
          kernel_shape=(3, 3),
          stride=s,
          with_bias=False,
          name='conv_{}'.format(i)))  # pytype: disable=not-callable
    if projection_shortcut:
      self._shortcut = hk.Conv2D(
          output_channels=num_filters,
          kernel_shape=(1, 1),
          stride=stride,
          with_bias=False,
          name='shortcut')  # pytype: disable=not-callable
    else:
      self._shortcut = None

  def __call__(self, inputs, **norm_kwargs):
    x = inputs
    orig_x = inputs
    for i, (bn, conv) in enumerate(zip(self._bn_modules, self._conv_modules)):
      x = bn(x, **norm_kwargs)
      x = self._activation(x)
      if self._shortcut is not None and i == 0:
        orig_x = x
      x = conv(x)
    if self._shortcut is not None:
      shortcut_x = self._shortcut(orig_x)
      x += shortcut_x
    else:
      x += orig_x
    return x


class WideResNet(hk.Module):
  """WideResNet designed for CIFAR-10."""

  def __init__(self,
               num_classes: int = 10,
               depth: int = 28,
               width: int = 10,
               split: str = 'all',
               activation: str = 'relu',
               bn_config: Optional[Dict[str, Any]] = None,
               name: Optional[str] = None):
    super(WideResNet, self).__init__(name=name)
    if (depth - 4) % 6 != 0:
      raise ValueError('depth should be 6n+4.')
    self._activation = getattr(jax.nn, activation)
    bn_config = dict(bn_config or {})
    bn_config.setdefault("decay_rate", 0.90)
    bn_config.setdefault("eps", 1e-5)
    bn_config.setdefault("create_scale", True)
    bn_config.setdefault("create_offset", True)
    self._split = split
    if split=='encoder' or split=='all':
      self._conv = hk.Conv2D(
          output_channels=16,
          kernel_shape=(3, 3),
          stride=1,
          with_bias=False,
          name='init_conv')  # pytype: disable=not-callable
      self._bn = hk.BatchNorm(
          name='batchnorm',
          **bn_config)

      blocks_per_layer = (depth - 4) // 6
      filter_sizes = [width * n for n in [16, 32, 64]]
      self._blocks = []
      for layer_num, filter_size in enumerate(filter_sizes):
        blocks_of_layer = []
        for i in range(blocks_per_layer):
          stride = 2 if (layer_num != 0 and i == 0) else 1
          projection_shortcut = (i == 0)
          blocks_of_layer.append(_WideResNetBlock(
              num_filters=filter_size,
              stride=stride,
              projection_shortcut=projection_shortcut,
              activation=self._activation,
              bn_config=bn_config,
              name='resnet_lay_{}_block_{}'.format(layer_num, i)))
        self._blocks.append(blocks_of_layer)
    if split=='decoder' or split=='all':
      self._linear = hk.Linear(
        num_classes,
        w_init=jnp.zeros,
        name='logits'
      )

  def __call__(self, inputs, is_training, test_local_stats=False):
    net = inputs
    split = self._split
    if split=='encoder' or split=='all':
        net = self._conv(net)

        # Blocks.
        for blocks_of_layer in self._blocks:
          for block in blocks_of_layer:
            net = block(net, is_training=is_training, test_local_stats=test_local_stats)
        net = self._bn(net, is_training=is_training, test_local_stats=test_local_stats)
        net = self._activation(net)

        net = jnp.mean(net, axis=[1, 2])
    if split=='decoder' or split=='all':
        net = self._linear(net)
    return net

class WideResNet28x10(WideResNet):
  def __init__(self,
               num_classes: int = 10,
               split: str = 'all',
               activation: str = 'relu',
               bn_config: Optional[Dict[str, Any]] = None,
               name: Optional[str] = None):
    super(WideResNet28x10, self).__init__(num_classes, 28, 10, split, activation, bn_config, name)

class WideResNet16x4(WideResNet):
  def __init__(self,
               num_classes: int = 10,
               split: str = 'all',
               activation: str = 'relu',
               bn_config: Optional[Dict[str, Any]] = None,
               name: Optional[str] = None):
    super(WideResNet16x4, self).__init__(num_classes, 16, 4, split, activation, bn_config, name)

class WideResNet22x4(WideResNet):
  def __init__(self,
               num_classes: int = 10,
               split: str = 'all',
               activation: str = 'relu',
               bn_config: Optional[Dict[str, Any]] = None,
               name: Optional[str] = None):
    super(WideResNet28x10, self).__init__(num_classes, 22, 4, split, activation, bn_config, name)

class WideResNet16x8(WideResNet):
  def __init__(self,
               num_classes: int = 10,
               split: str = 'all',
               activation: str = 'relu',
               bn_config: Optional[Dict[str, Any]] = None,
               name: Optional[str] = None):
    super(WideResNet16x8, self).__init__(num_classes, 16, 8, split, activation, bn_config, name)
