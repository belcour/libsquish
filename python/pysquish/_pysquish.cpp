#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "squish/squish.h"

namespace py = pybind11;

static py::bytes compress_rgba_to_dxt_bytes(py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> rgba,
                                            int flags) {
    if (rgba.ndim() != 3 || rgba.shape(2) != 4) {
        throw std::invalid_argument("Expected an array of shape (H, W, 4) with RGBA uint8 pixels.");
    }

    const int height = static_cast<int>(rgba.shape(0));
    const int width = static_cast<int>(rgba.shape(1));
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Image dimensions must be positive.");
    }

    // libsquish expects an RGBA8 contiguous buffer and allows arbitrary image sizes.
    const int pitch = width * 4;

    const int storage_i = squish::GetStorageRequirements(width, height, flags);
    if (storage_i <= 0) {
        throw std::runtime_error("Computed compressed storage size is invalid.");
    }

    std::vector<squish::u8> out(static_cast<size_t>(storage_i));
    squish::CompressImage(reinterpret_cast<squish::u8 const*>(rgba.data()), width, height, pitch,
                           out.data(), flags, /*metric=*/nullptr);

    return py::bytes(reinterpret_cast<char const*>(out.data()), out.size());
}

PYBIND11_MODULE(_pysquish, m) {
    m.doc() = "pysquish - Python bindings for libsquish";
    m.def("compress_rgba_to_dxt_bytes", &compress_rgba_to_dxt_bytes,
          py::arg("rgba"), py::arg("flags") = 0,
          "Compress an RGBA8 image to DXT/BC blocks and return the raw block bytes.");
}

