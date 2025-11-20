import os
from glob import glob
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


CUDA_INCLUDE_CANDIDATES = [
    os.environ.get("CUDA_HOME"),
    os.environ.get("CUDA_PATH"),
    "/usr/local/cuda",
]


def cuda_available() -> bool:
    """Check for a CUDA toolkit installation by probing for cuda_runtime.h."""

    for path in CUDA_INCLUDE_CANDIDATES:
        if not path:
            continue
        candidate = Path(path) / "include" / "cuda_runtime.h"
        if candidate.exists():
            return True
    return False


class build_ext_optional(build_ext):
    """Custom build_ext that can be skipped via WARPDB_SKIP_CUDA."""

    def run(self):
        if os.environ.get("WARPDB_SKIP_CUDA"):
            self.distribution.ext_modules = []
            self.extensions = []
            return

        if not cuda_available():
            raise RuntimeError(
                "CUDA toolkit not found. Install CUDA or set WARPDB_SKIP_CUDA=1"
                " to install without GPU extensions."
            )

        super().run()

include_files = glob('include/*.hpp') + glob('include/*.h')
data_files = ['data/test.csv', 'data/test.json', 'data/malformed.json']

ext_modules = [
    Pybind11Extension(
        'pywarpdb',
        [
            'bindings/python/pywarpdb.cpp',
            'src/warpdb.cpp',
            'src/csv_loader.cpp',
            'src/json_loader.cpp',
            'src/expression.cpp',
            'src/jit.cpp',
            'src/optimizer.cpp',
            'src/arrow_utils.cpp',
        ],
        include_dirs=['include'],
        extra_link_args=['-lcudart', '-lnvrtc', '-lcuda'],
        cxx_std=17,
    )
]

setup(
    name='warpdb',
    version='0.1.0',
    description='Python bindings for WarpDB',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext_optional},
    setup_requires=['pybind11>=2.6'],
    data_files=[('include', include_files), ('data', data_files)],
    zip_safe=False,
)
