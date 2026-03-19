from __future__ import annotations

import numpy as np

from ._pysquish import compress_rgba_to_dxt_bytes
from .dds import write_dds


# Keep these in sync with squish.h.
kDxt1 = (1 << 0)
kDxt3 = (1 << 1)
kDxt5 = (1 << 2)
kBc4 = (1 << 3)
kBc5 = (1 << 4)

kColourClusterFit = (1 << 5)
kColourRangeFit = (1 << 6)
kColourIterativeClusterFit = (1 << 8)
kWeightColourByAlpha = (1 << 7)


def compress(rgba: np.ndarray, filepath, flags: int = kDxt1 | kColourClusterFit) -> str:
    """
    Compress an RGBA image using libsquish and write it to `filepath` as a `.dds` file.

    Parameters
    ----------
    rgba:
        Numpy array with shape `(H, W, 4)` (RGBA) or `(H, W, 3)` (RGB; alpha assumed to be 255).
        Supported dtypes: `uint8`, or numeric types convertible to `uint8`.
        If float input is provided, values are assumed to be in `[0, 1]`.
    filepath:
        Output `.dds` path.
    flags:
        libsquish compression flags (DXT1/DXT3/DXT5/BC4/BC5 + quality options).
        Defaults to `kDxt1 | kColourClusterFit`.
    """
    arr = np.asarray(rgba)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError("Expected `rgba` with shape (H, W, 4) or (H, W, 3).")

    if arr.dtype == np.uint8:
        arr_u8 = np.ascontiguousarray(arr)
    else:
        if np.issubdtype(arr.dtype, np.floating):
            arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
        arr_u8 = np.ascontiguousarray(arr_u8)

    if arr_u8.shape[-1] == 3:
        # Failsafe for RGB inputs: append opaque alpha channel.
        h, w, _ = arr_u8.shape
        rgba_u8 = np.empty((h, w, 4), dtype=np.uint8)
        rgba_u8[..., :3] = arr_u8
        rgba_u8[..., 3] = 255
    else:
        rgba_u8 = arr_u8

    height, width = int(rgba_u8.shape[0]), int(rgba_u8.shape[1])

    blocks = compress_rgba_to_dxt_bytes(rgba_u8, int(flags))
    write_dds(filepath, blocks, width=width, height=height, flags=int(flags))
    return str(filepath)


__all__ = [
    "compress",
    "kDxt1",
    "kDxt3",
    "kDxt5",
    "kBc4",
    "kBc5",
    "kColourClusterFit",
    "kColourRangeFit",
    "kColourIterativeClusterFit",
    "kWeightColourByAlpha",
]

