"""Datasets: raw uint8 images + a small `DatasetSpec` that makes the harness dataset-agnostic.

Images stay uint8 and flat, `(N, n_pixels)` -- that is what the circuit gets, so any float,
threshold or normalisation is part of the model and lands in its gate count. numpy only.

`DatasetSpec` is the one object every emitter and the simulator read, so MNIST (784x8) and CIFAR10
(3072x8 RGB) go through identical code. `load(name)` returns a `Dataset` carrying its spec.
"""

from __future__ import annotations

import gzip
import struct
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class DatasetSpec:
    """The fixed circuit interface for one dataset:

        module top (input [port_bits-1:0] pix, output [cls_bits-1:0] cls);

    `pix[pixel_bits*p +: pixel_bits]` is byte `p`; `cls` is the predicted class.
    """

    name: str
    n_pixels: int      # input bytes (784 MNIST, 3072 CIFAR10 RGB)
    pixel_bits: int    # bits per byte (8)
    n_classes: int     # 10

    @property
    def port_bits(self) -> int:
        return self.n_pixels * self.pixel_bits

    @property
    def cls_bits(self) -> int:
        return max(1, (self.n_classes - 1).bit_length())


@dataclass
class Dataset:
    spec: DatasetSpec
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def _split(train_x, train_y, test_x, test_y, *, spec, val_size, seed=0) -> Dataset:
    """Cut a fixed val slice off the official training set with a seeded permutation."""
    perm = np.random.default_rng(seed).permutation(len(train_x))
    val, tr = perm[:val_size], perm[val_size:]
    return Dataset(spec, train_x[tr], train_y[tr], train_x[val], train_y[val], test_x, test_y)


def to_bits(pix: np.ndarray, spec: DatasetSpec) -> np.ndarray:
    """`(N, n_pixels)` uint8 -> `(N, port_bits)` uint8 bits, in port order: bit `pixel_bits*p + k`
    is bit `k` (LSB first) of byte `p`, i.e. exactly `pix[pixel_bits*p +: pixel_bits]`."""
    if pix.shape[1] != spec.n_pixels:
        raise ValueError(f"{spec.name}: expected {spec.n_pixels} bytes/image, got {pix.shape[1]}")
    bits = (pix[:, :, None] >> np.arange(spec.pixel_bits, dtype=np.uint8)) & 1
    return bits.reshape(len(pix), spec.port_bits).astype(np.uint8)


# --- MNIST --------------------------------------------------------------------------------------

MNIST = DatasetSpec("mnist", 784, 8, 10)
_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def _idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 0x803:
        h, w = struct.unpack(">II", raw[8:16])
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, h * w).copy()
    if magic == 0x801:
        return np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    raise ValueError(f"{path}: bad idx magic {magic:#x}")


def _load_mnist() -> Dataset:
    d = DATA_ROOT / "mnist"
    d.mkdir(parents=True, exist_ok=True)
    for f in _MNIST_FILES:
        if not (d / f).exists():
            print(f"downloading {f} ...", flush=True)
            urllib.request.urlretrieve(_MNIST_MIRROR + f, d / f)
    return _split(_idx(d / _MNIST_FILES[0]), _idx(d / _MNIST_FILES[1]),
                  _idx(d / _MNIST_FILES[2]), _idx(d / _MNIST_FILES[3]), spec=MNIST, val_size=6000)


# --- CIFAR10 (raw RGB bytes, interleaved R,G,B per pixel: byte 3*q+c) ---------------------------

CIFAR10 = DatasetSpec("cifar10", 3072, 8, 10)
_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
_CIFAR_INNER = "cifar-10-batches-bin"
_CIFAR_REC = 1 + 3072  # label byte + 3072 planar image bytes (1024 R, 1024 G, 1024 B)


def _cifar_bins(paths) -> tuple[np.ndarray, np.ndarray]:
    raw = np.concatenate([np.frombuffer(p.read_bytes(), np.uint8) for p in paths]).reshape(-1, _CIFAR_REC)
    y = raw[:, 0].astype(np.int64)
    planar = raw[:, 1:].reshape(-1, 3, 1024)                # (N, channel, pixel)
    x = planar.transpose(0, 2, 1).reshape(-1, 3072).copy()  # interleave -> byte 3*q+c
    return x, y


def _load_cifar10() -> Dataset:
    d = DATA_ROOT / "cifar10"
    d.mkdir(parents=True, exist_ok=True)
    inner = d / _CIFAR_INNER
    if not inner.exists():
        tar = d / "cifar-10-binary.tar.gz"
        if not tar.exists():
            print("downloading cifar-10-binary.tar.gz (~170 MB) ...", flush=True)
            urllib.request.urlretrieve(_CIFAR_URL, tar)
        with tarfile.open(tar, "r:gz") as t:
            t.extractall(d)  # noqa: S202 -- trusted, well-known archive
    tr = _cifar_bins([inner / f"data_batch_{i}.bin" for i in range(1, 6)])
    te = _cifar_bins([inner / "test_batch.bin"])
    return _split(*tr, *te, spec=CIFAR10, val_size=5000)


# --- registry -----------------------------------------------------------------------------------

SPECS = {"mnist": MNIST, "cifar10": CIFAR10}
_LOADERS = {"mnist": _load_mnist, "cifar10": _load_cifar10}
NAMES = tuple(_LOADERS)


def spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(NAMES)}")
    return SPECS[name]


def load(name: str) -> Dataset:
    if name not in _LOADERS:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(NAMES)}")
    return _LOADERS[name]()
