from audioop import bias
import jax
import equinox as eqx
from equinox import nn
from numpy import pad

class BacisBlock2D(eqx.Module):
    conv1: eqx.Module
    conv2: eqx.Module
    # pool: eqx.nn.Pool

    def __init__(self, in_planes, planes, stride=1, group_norm=False, *, key) -> None:
        super().__init__()
        c1k, c2k = jax.random.split(key)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=(1, 1), use_bias=False, key=c1k)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                            stride=1, padding=(1, 1), use_bias=False, key=c2k)
        # self.pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

    def __call__(self, h, *, key = None):
        out = jax.nn.relu(self.conv1(h))
        out = self.conv2(out)
        out += h
        out = jax.nn.relu(out)
        # out = self.pool(out)
        return out

class PoolBlock2D(eqx.Module):
    conv1: eqx.Module
    conv2: eqx.Module
    pool: eqx.nn.Pool

    def __init__(self, in_planes, planes, stride=1, group_norm=False, *, key) -> None:
        super().__init__()
        c1k, c2k = jax.random.split(key)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=(1, 1), use_bias=False, key=c1k)
        # self.gn1 = nn.GroupNorm(4, planes, affine=False) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                            stride=1, padding=(1, 1), use_bias=False, key=c2k)
        self.pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

    def __call__(self, h, *, key = None):
        out = jax.nn.relu(self.conv1(h))
        out = self.conv2(out)
        out += h
        # out = jax.nn.relu(out)
        out = self.pool(out)
        return out

class RecurModel(eqx.Module):
    proj: eqx.Module
    iter_block: eqx.Module
    head: eqx.Module

    def __init__(self, in_channels: int, width: int, *, key) -> None:
        super().__init__()
        keys = jax.random.split(key, 6)
        self.proj = nn.Conv2d(in_channels, width, 3, stride=(1, 1), padding=(1, 1), use_bias=False, key=keys[0])
        self.iter_block = nn.Sequential([BacisBlock2D(width, width, key=keys[1]), BacisBlock2D(width, width, key=keys[2])])
        self.head = [
                nn.Conv2d(width, 32, kernel_size=3, stride=1, padding=1, use_bias=False, key=keys[3]), 
                nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1,  use_bias=False, key=keys[4]),
                nn.Conv2d(8, 2, kernel_size=3, stride=1, padding=1,  use_bias=False, key=keys[5]),
            ]
    
    def __call__(self, input, iters=1):
        h = jax.nn.relu(self.proj(input))
        # h = self.iter_block(h)
        h = jax.lax.fori_loop(0, iters, lambda i, h: self.iter_block(h), h)
        out = jax.nn.relu(self.head[0](h))
        out = jax.nn.relu(self.head[1](out))
        out = self.head[2](out)
        return out


