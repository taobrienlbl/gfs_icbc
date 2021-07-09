CC=gcc
FC=gfortran
F2PY=f2py
F2PYFLAGS=--opt="-O3" #--fcompiler="intele"

FFLAGS=-O3 -fPIC

SRC = \
	ftnvectorint2p.f90 \
	int2p_dp.f

SOSRC = \
	ftnvectorint2p.f90

OBJ := $(addsuffix .o, $(basename $(SRC)))
SOFILES := $(addsuffix .so, $(basename $(SOSRC)))

all: main

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

%.o: %.c
	${CC} ${CFLAGS} -c $< -o $@

ftnvectorint2p.so: ftnvectorint2p.f90 int2p_dp.o
	${F2PY} -c ${F2PYFLAGS} -m ftnvectorint2p $^

main: $(OBJ) $(SOFILES)

clean:
	-rm -f ./*.o ./*.mod ./*.so

