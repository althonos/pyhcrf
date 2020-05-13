import configparser
import sys

import numpy
import setuptools
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.sdist import sdist as _sdist


class sdist(_sdist):
    """A `sdist` that generates a `pyproject.toml` on the fly.
    """

    def run(self):
        # build `pyproject.toml` from `setup.cfg`
        c = configparser.ConfigParser()
        c.add_section("build-system")
        c.set("build-system", "requires", str(self.distribution.setup_requires))
        c.set("build-system", 'build-backend', '"setuptools.build_meta"')
        with open("pyproject.toml", "w") as pyproject:
            c.write(pyproject)
        # run the rest of the packaging
        _sdist.run(self)


class build_ext(_build_ext):
    """A `build_ext` that disables optimizations if compiled in debug mode.
    """

    def build_extension(self, ext):
        if self.debug:
            ext.extra_compile_args.append("-O0")
            if sys.implementation.name == "cpython":
                ext.define_macros.append(("CYTHON_TRACE_NOGIL", 1))
        _build_ext.build_extension(self, ext)


setuptools.setup(
    cmdclass=dict(sdist=sdist, build_ext=build_ext),
    ext_modules=cythonize(
        setuptools.Extension(
            "pyhcrf._algorithms",
            ["pyhcrf/_algorithms.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-Wall"],
        ),
        annotate=True,
    ),
)
