# Modified by Trung Trinh from: https://github.com/DensoITLab/featurePI/blob/main/lib/models/resnet.py

from functools import partial
from typing import Any, Callable, Sequence, Tuple, Optional

from flax import linen as nn
import jax
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlockV1(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    projection: bool
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            if self.projection:
                residual = self.conv(
                    self.filters, (1, 1), self.strides, name="conv_proj"
                )(residual)
                residual = self.norm(name="norm_proj")(residual)
            else:
                residual = jnp.pad(
                    residual[:, :: self.strides[0], :: self.strides[1], :],
                    ((0, 0), (0, 0), (0, 0), (self.filters // 4, self.filters // 4)),
                    "constant",
                )

        return self.act(residual + y)
    
class ResNetBlockV2(nn.Module):
    """ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    projection: bool
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        x = self.norm()(x)
        x = residual = self.act(x)
        x = self.conv(self.filters, (3, 3), self.strides)(x)        
        x = self.norm()(x)
        x = self.act(x)
        x = self.conv(self.filters, (3, 3))(x)
        if residual.shape != x.shape:
            if self.projection:
                residual = self.conv(
                    self.filters, (1, 1), self.strides, name="conv_proj"
                )(residual)
            else:
                residual = jnp.pad(
                    residual[:, :: self.strides[0], :: self.strides[1], :],
                    ((0, 0), (0, 0), (0, 0), (self.filters // 4, self.filters // 4)),
                    "constant",
                )

        return x + residual


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    projection: bool
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNet(nn.Module):

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    low_res: bool = True
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    projection: bool = True
    resnet_v2: bool = False
    bn_axis_name: Optional[str] = 'batch'

    def encode(self, x, train):
        conv = partial(
            nn.Conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.variance_scaling(2.0, "fan_out", "normal"),
        )
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5,
            dtype=self.dtype,
            axis_name=self.bn_axis_name
        )

        x = conv(
            self.num_filters,
            (3, 3) if self.low_res else (7, 7),
            (1, 1) if self.low_res else (2, 2),
            padding="SAME",
            name="conv_init",
        )(x)
        if not self.resnet_v2:
          x = norm(name="bn_init")(x)
          x = nn.relu(x)
        if not self.low_res:
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2**i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act,
                    projection=self.projection,
                )(x)
        if self.resnet_v2:
            x = norm(name='bn_final')(x)
            x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2), dtype=self.dtype)
        return x

    def classify(self, x, train):
        x = nn.Dense(
            self.num_classes, dtype=self.dtype, kernel_init=dense_layer_init_fn
        )(x)
        return x

    @nn.compact
    def __call__(self, x, train):
        x = self.encode(x, train)
        x = self.classify(x, train)
        return x


def dense_layer_init_fn(
    key: jnp.ndarray, shape: Tuple[int, int], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """Initializer for the final dense layer.
    Args:
        key: PRNG key to use to sample the weights.
        shape: Shape of the tensor to initialize.
        dtype: Data type of the tensor to initialize.
    Returns:
        The initialized tensor.
    """
    num_units_out = shape[1]
    unif_init_range = 1.0 / (num_units_out) ** (0.5)
    return jax.random.uniform(key, shape, dtype, -1) * unif_init_range


ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlockV1)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlockV1)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock)

PreActResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlockV2, resnet_v2=True)
PreActResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3], block_cls=ResNetBlockV2, resnet_v2=True)
