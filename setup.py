from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

include_files = glob('include/*.hpp') + glob('include/*.h')
data_files = ['data/test.csv', 'data/test.json']

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
    cmdclass={'build_ext': build_ext},
    data_files=[('include', include_files), ('data', data_files)],
    zip_safe=False,
)
