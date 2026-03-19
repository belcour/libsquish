from pathlib import Path

from setuptools import Extension, find_packages, setup


def _has_flag(flag: str) -> bool:
    # Very small helper; avoids sprinkling platform-specific logic.
    import sys

    return flag in sys.argv


def main():
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    # Build-time deps (installed via pyproject).
    import numpy as np
    import pybind11

    libsquish_sources = [
        repo_root / "lib/src/alpha.cpp",
        repo_root / "lib/src/clusterfit.cpp",
        repo_root / "lib/src/colourblock.cpp",
        repo_root / "lib/src/colourfit.cpp",
        repo_root / "lib/src/colourset.cpp",
        repo_root / "lib/src/maths.cpp",
        repo_root / "lib/src/rangefit.cpp",
        repo_root / "lib/src/singlecolourfit.cpp",
        repo_root / "lib/src/squish.cpp",
    ]

    ext_sources = libsquish_sources + [
        here / "pysquish" / "_pysquish.cpp",
    ]

    extra_compile_args = ["-O3", "-std=c++11"]
    if _has_flag("--debug"):
        extra_compile_args = ["-O0", "-g", "-std=c++11"]

    ext = Extension(
        name="pysquish._pysquish",
        sources=[str(p) for p in ext_sources],
        include_dirs=[
            str(repo_root / "lib/include"),
            str(repo_root / "lib/src"),
            # CMake normally generates `squish/squish_export.h`; provide a stub
            # for building the extension in isolation.
            str(here / "compat"),
            pybind11.get_include(),
            np.get_include(),
        ],
        language="c++",
        extra_compile_args=extra_compile_args,
    )

    setup(
        name="pysquish",
        version="0.1.0",
        description="Python bindings for libsquish (BC/DXT compression) with DDS output",
        packages=find_packages(where=str(here)),
        package_dir={"": str(here)},
        ext_modules=[ext],
        zip_safe=False,
    )


if __name__ == "__main__":
    main()

