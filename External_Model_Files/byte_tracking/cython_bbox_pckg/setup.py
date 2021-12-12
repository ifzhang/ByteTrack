from __future__ import print_function

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension
from setuptools import setup

import numpy as np


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        name='cython_bbox',
        sources=['cython_bbox.pyx'],
        extra_compile_args = {'gcc': ['/Qstd=c99']},
        include_dirs=[numpy_include]
    )
]

setup(
    name='cython_bbox',
    ext_modules=cythonize(ext_modules),
    version = '0.1.3',
    description = 'Standalone cython_bbox',
    keywords = ['cython_bbox']
)