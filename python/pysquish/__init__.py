from __future__ import annotations

import io
import struct
import numpy as np

from ._pysquish import compress_rgba_to_dxt_bytes  # type: ignore[import-not-found]
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


def _as_rgba_u8(rgba: np.ndarray) -> np.ndarray:
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
        h, w, _ = arr_u8.shape
        rgba_u8 = np.empty((h, w, 4), dtype=np.uint8)
        rgba_u8[..., :3] = arr_u8
        rgba_u8[..., 3] = 255
        return rgba_u8

    return arr_u8


def _dds_bytes_from_blocks(rgba_to_blocks_bytes: bytes, width: int, height: int, flags: int) -> bytes:
    """
    Create an in-memory DDS container for DXT/BC blocks produced by libsquish.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive.")
    if not isinstance(rgba_to_blocks_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("Compressed data must be bytes-like.")

    # Keep these in sync with `python/pysquish/dds.py`.
    def _fourcc(code: str) -> int:
        if len(code) != 4:
            raise ValueError("FourCC code must be 4 characters.")
        return int.from_bytes(code.encode("ascii"), "little", signed=False)

    def _method_from_flags(f: int) -> int:
        method_mask = (kDxt1 | kDxt3 | kDxt5 | kBc4 | kBc5)
        method = f & method_mask
        return method if method != 0 else kDxt1

    def _fourcc_and_blocksize(f: int):
        method = _method_from_flags(f)
        if method == kDxt1:
            return _fourcc("DXT1"), 8
        if method == kDxt3:
            return _fourcc("DXT3"), 16
        if method == kDxt5:
            return _fourcc("DXT5"), 16
        if method == kBc4:
            return _fourcc("ATI1"), 8
        if method == kBc5:
            return _fourcc("ATI2"), 16
        raise ValueError(f"Unsupported compression method flags: {f}")

    def _expected_data_size(w: int, h: int, f: int) -> int:
        _, bytes_per_block = _fourcc_and_blocksize(f)
        blockcount = ((w + 3) // 4) * ((h + 3) // 4)
        return blockcount * bytes_per_block

    data = bytes(rgba_to_blocks_bytes)
    expected = _expected_data_size(int(width), int(height), int(flags))
    if len(data) != expected:
        raise ValueError(f"Compressed data size mismatch: got {len(data)}, expected {expected}.")

    fourcc, _bytes_per_block = _fourcc_and_blocksize(int(flags))

    # DDS constants
    DDSD_CAPS = 0x00000001
    DDSD_HEIGHT = 0x00000002
    DDSD_WIDTH = 0x00000004
    DDSD_PIXELFORMAT = 0x00001000
    DDSD_LINEARSIZE = 0x00080000
    DDSCAPS_TEXTURE = 0x00001000
    DDPF_FOURCC = 0x00000004

    header = struct.pack(
        "<31I",
        124,
        DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE,
        int(height),
        int(width),
        len(data),
        0,
        0,
        *([0] * 11),
        32,
        DDPF_FOURCC,
        fourcc,
        0,
        0,
        0,
        0,
        0,
        DDSCAPS_TEXTURE,
        0,
        0,
        0,
        0,
    )

    return b"DDS " + header + data


def compress_to_array(rgba: np.ndarray, flags: int = kDxt1 | kColourClusterFit) -> np.ndarray:
    """
    Compress an image with libsquish and return the decoded result as an RGBA float array.

    The returned array has shape (H, W, 4), dtype float32, and values in [0, 1].
    This round-trips through an in-memory DDS container to let Pillow decode the DXT blocks.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow is required for `compress_to_array()` (pip install pillow).") from e

    rgba_u8 = _as_rgba_u8(rgba)
    height, width = int(rgba_u8.shape[0]), int(rgba_u8.shape[1])

    blocks = compress_rgba_to_dxt_bytes(rgba_u8, int(flags))
    dds_bytes = _dds_bytes_from_blocks(blocks, width=width, height=height, flags=int(flags))

    im = Image.open(io.BytesIO(dds_bytes))
    im.load()
    rgba_out_u8 = np.asarray(im.convert("RGBA"), dtype=np.uint8)
    return rgba_out_u8.astype(np.float32) / 255.0


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
    rgba_u8 = _as_rgba_u8(rgba)

    height, width = int(rgba_u8.shape[0]), int(rgba_u8.shape[1])

    blocks = compress_rgba_to_dxt_bytes(rgba_u8, int(flags))
    write_dds(filepath, blocks, width=width, height=height, flags=int(flags))
    return str(filepath)


__all__ = [
    "compress",
    "compress_to_array",
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

