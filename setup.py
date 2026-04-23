from glob import glob
import os
from pathlib import Path
from urllib.request import urlopen

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


NLOHMANN_JSON_VERSION = 'v3.2.0'
NLOHMANN_JSON_URLS = [
    f'https://raw.githubusercontent.com/nlohmann/json/{NLOHMANN_JSON_VERSION}/single_include/nlohmann/json.hpp',
    f'https://raw.githubusercontent.com/nlohmann/json/{NLOHMANN_JSON_VERSION}/include/nlohmann/json.hpp',
]


def ensure_nlohmann_json_header(base_dir: Path) -> str:
    include_dir = base_dir / 'nlohmann_json' / NLOHMANN_JSON_VERSION / 'include'
    header_path = include_dir / 'nlohmann' / 'json.hpp'
    if header_path.exists():
        return str(include_dir)

    header_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for url in NLOHMANN_JSON_URLS:
        try:
            with urlopen(url) as response:
                header_path.write_bytes(response.read())
            return str(include_dir)
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(
        f'Unable to download nlohmann/json.hpp for {NLOHMANN_JSON_VERSION}: {last_error}'
    )


class WarpDBBuildExt(build_ext):
    def build_extensions(self):
        include_dir = ensure_nlohmann_json_header(Path(__file__).resolve().parent / '.build_deps')
        for ext in self.extensions:
            if include_dir not in ext.include_dirs:
                ext.include_dirs.append(include_dir)

        self._configure_cuda_compiler()
        super().build_extensions()

    def _configure_cuda_compiler(self):
        if not hasattr(self.compiler, 'compiler_so'):
            return

        if '.cu' not in self.compiler.src_extensions:
            self.compiler.src_extensions.append('.cu')

        default_compiler_so = self.compiler.compiler_so
        super_compile = self.compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            is_cuda = os.path.splitext(src)[1] == '.cu'

            if isinstance(extra_postargs, dict):
                cxx_postargs = extra_postargs.get('cxx', [])
                nvcc_postargs = extra_postargs.get('nvcc', [])
            else:
                cxx_postargs = extra_postargs
                nvcc_postargs = extra_postargs

            try:
                if is_cuda:
                    self.compiler.set_executable('compiler_so', os.environ.get('NVCC', 'nvcc'))
                    postargs = ['-std=c++17', *nvcc_postargs]
                else:
                    postargs = cxx_postargs

                super_compile(obj, src, ext, cc_args, postargs, pp_opts)
            finally:
                self.compiler.compiler_so = default_compiler_so

        self.compiler._compile = _compile


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
            'src/jit.cu',
            'src/multi_gpu_utils.cpp',
            'src/optimizer.cpp',
            'src/arrow_utils.cpp',
        ],
        include_dirs=['include'],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--compiler-options', '-fPIC'],
        },
        extra_link_args=['-lcudart', '-lnvrtc', '-lcuda'],
        cxx_std=17,
    )
]

setup(
    name='warpdb',
    version='0.1.0',
    description='Python bindings for WarpDB',
    ext_modules=ext_modules,
    cmdclass={'build_ext': WarpDBBuildExt},
    data_files=[('include', include_files), ('data', data_files)],
    zip_safe=False,
)
