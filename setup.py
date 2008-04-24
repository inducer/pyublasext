#!/usr/bin/env python
# -*- coding: latin-1 -*-

import os
import os.path
import sys

def get_config_schema():
    from aksetup_helper import ConfigSchema, Option, \
            IncludeDir, LibraryDir, Libraries, \
            Switch, StringListOption

    return ConfigSchema([
        IncludeDir("BOOST", []),
        LibraryDir("BOOST", []),
        Libraries("BOOST_PYTHON", ["boost_python-gcc42-mt"]),

        IncludeDir("BOOST_BINDINGS", []),

        Switch("HAVE_BLAS", False, "Whether to build with support for BLAS"),
        LibraryDir("BLAS", []),
        Libraries("BLAS", ["blas"]),

        Switch("HAVE_LAPACK", False, "Whether to build with support for LAPACK"),
        LibraryDir("LAPACK", []),
        Libraries("LAPACK", ["blas"]),

        Switch("COMPILE_DASKR", True, "Whether to build (with) DASKR"),
        Switch("COMPILE_XERBLA", False, 
            "Whether to compile and add our own XERBLA routine."
            "ATLAS LAPACK does not have one."),

        Switch("HAVE_ARPACK", False, "Whether to build with support for BLAS"),
        LibraryDir("ARPACK", []),
        Libraries("ARPACK", ["arpack"]),

        Switch("HAVE_UMFPACK", False, "Whether to build with support for UMFPACK"),
        IncludeDir("UMFPACK", ["/usr/include/suitesparse"]),
        LibraryDir("UMFPACK", []),
        Libraries("UMFPACK", ["umfpack", "amd"]),

        StringListOption("CXXFLAGS", [], 
            help="Any extra C++ compiler options to include"),
        StringListOption("LDFLAGS", [], 
            help="Any extra linker options to include"),
        ])




def main():
    import glob
    import os
    from aksetup_helper import hack_distutils, get_config, setup, \
            NumpyExtension

    hack_distutils()
    conf = get_config(get_config_schema())

    # These are in Fortran. No headers available.
    conf["BLAS_INC_DIR"] = []
    conf["LAPACK_INC_DIR"] = []
    conf["ARPACK_INC_DIR"] = []
    conf["DASKR_INC_DIR"] = []
    conf["XERBLA_INC_DIR"] = []

    conf["DASKR_LIB_DIR"] = ["fortran/daskr"]
    conf["DASKR_LIBNAME"] = ["daskr"]
    conf["XERBLA_LIB_DIR"] = ["fortran/xerbla"]
    conf["XERBLA_LIBNAME"] = ["xerbla"]

    if conf["COMPILE_DASKR"]:
        os.system("cd fortran/daskr; ./build.sh")
    if conf["COMPILE_XERBLA"]:
        os.system("cd fortran/xerbla; ./build.sh")

    INCLUDE_DIRS = ["src/cpp"] + conf["BOOST_INC_DIR"]

    LIBRARY_DIRS = conf["BOOST_LIB_DIR"]
    LIBRARIES = conf["BOOST_PYTHON_LIBNAME"]

    OP_EXTRA_INCLUDE_DIRS = conf["BOOST_BINDINGS_INC_DIR"]
    OP_EXTRA_LIBRARY_DIRS = []
    OP_EXTRA_LIBRARIES = []

    conf["USE_XERBLA"] = conf["COMPILE_XERBLA"]
    conf["USE_BLAS"] = conf["HAVE_BLAS"]
    conf["USE_LAPACK"] = conf["HAVE_LAPACK"] and conf["HAVE_BLAS"]
    conf["USE_ARPACK"] = conf["HAVE_ARPACK"] and conf["USE_LAPACK"]
    conf["USE_UMFPACK"] = conf["USE_BLAS"] and conf["HAVE_UMFPACK"]
    conf["USE_DASKR"] = conf["USE_LAPACK"] and conf["COMPILE_DASKR"]

    if conf["HAVE_LAPACK"] and not conf["USE_LAPACK"]:
        print "*** LAPACK disabled because BLAS is missing"
    if conf["HAVE_ARPACK"] and not conf["USE_LAPACK"]:
        print "*** ARPACK disabled because LAPACK is not usable/missing"
    if conf["HAVE_UMFPACK"] and not conf["USE_UMFPACK"]:
        print "*** UMFPACK disabled because BLAS is missing"

    OP_EXTRA_DEFINES = { "PYUBLAS_HAVE_BOOST_BINDINGS":1 }

    def handle_component(comp):
        if conf["USE_"+comp]:
            OP_EXTRA_DEFINES["USE_"+comp] = 1
            OP_EXTRA_INCLUDE_DIRS.extend(conf[comp+"_INC_DIR"])
            OP_EXTRA_LIBRARY_DIRS.extend(conf[comp+"_LIB_DIR"])
            OP_EXTRA_LIBRARIES.extend(conf[comp+"_LIBNAME"])

    handle_component("ARPACK")
    handle_component("UMFPACK")
    handle_component("DASKR")
    handle_component("LAPACK")
    handle_component("BLAS")
    handle_component("XERBLA")

    setup(name="PyUblasExt",
          version="0.92.4",
          description="Added functionality for PyUblas",
          long_description="""
          PyUblasExt is a companion to 
          `PyUblas <http://mathema.tician.de/software/pyublas>`_
          and exposes a variety of useful additions to it:

          * A cross-language "operator" class for building matrix-free algorithms
          * CG and BiCGSTAB linear solvers that use this operator class
          * An `ARPACK <http://mathema.tician.de/software/arpack>`_ interface that also uses this operator class
          * An UMFPACK interface for PyUblas's sparse matrices
          * An interface to the `DASKR <http://www.netlib.org/ode/>`_ ODE solver.

          Please refer to the `PyUblas build documentation
          <http://tiker.net/doc/pyublas>`_ for build instructions.
          """,
          author=u"Andreas Kloeckner",
          author_email="inform@tiker.net",
          license = "BSD",
          url="http://mathema.tician.de/software/pyublas/pyublasext",
          classifiers=[
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: POSIX',
              'Programming Language :: Python',
              'Programming Language :: C++',
              'Topic :: Scientific/Engineering',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Utilities',
              ],

          # dependencies
          setup_requires=[
              "PyUblas>=0.92.4",
              ],
          install_requires=[
              "PyUblas>=0.92.4",
              ],

          packages=["pyublasext"],
          zip_safe=False,
          package_dir={"pyublasext": "src/python"},
          ext_package="pyublasext",
          ext_modules=[ NumpyExtension( "_internal", 
              [
                  "src/wrapper/operation.cpp",
                  "src/wrapper/op_daskr.cpp",
                  ],
              define_macros=list(OP_EXTRA_DEFINES.iteritems()),
              include_dirs=INCLUDE_DIRS + OP_EXTRA_INCLUDE_DIRS,
              library_dirs=LIBRARY_DIRS + OP_EXTRA_LIBRARY_DIRS,
              libraries=LIBRARIES + OP_EXTRA_LIBRARIES,
              extra_compile_args=conf["CXXFLAGS"],
              extra_link_args=conf["LDFLAGS"],
              ), ],
          data_files=[("include/pyublasext", glob.glob("src/cpp/pyublasext/*.hpp"))],
         )




if __name__ == '__main__':
    main()
