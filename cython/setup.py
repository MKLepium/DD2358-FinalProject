from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        
        "ray_tracing",
        ["ray_tracing.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='ray_tracing',
    ext_modules=cythonize(ext_modules),
)
