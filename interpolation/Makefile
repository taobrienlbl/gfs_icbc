.PHONY: default
default: build

PYTHON = python
PYTHON_INCLUDE = ${shell ${PYTHON} -c 'from distutils import sysconfig; print( sysconfig.get_python_inc() )'}
NUMPY_INCLUDE = ${shell ${PYTHON} -c 'import numpy; print( numpy.get_include() )'}

TIMEIT = ${shell ${PYTHON} -c 'import timeit; print(timeit.__file__)'}

FORTMODDIR=fortranModuleSource

CC = gcc -shared -fPIC
CFLAGS = -shared -fPIC
SO = ${shell ${PYTHON} -c 'import imp; print (imp.get_suffixes()[0][0])'}

OBJ := $(addsuffix .o, $(basename $(SRC)))
SOFILES := $(addsuffix .so, $(basename $(SOSRC)))

.PHONY: build
build: fsofiles

.PHONY: fsofiles
fsofiles:
	cd $(FORTMODDIR); make all
#	cp $(FORTMODDIR)/*.so .

.PHONY: clean
clean:
	cd $(FORTMODDIR); make clean

