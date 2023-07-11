# Intel(R) C / C++ Compiler Settings

# source $(ICCDIR)/bin/iccvars.sh

ICCDIR          = /usr/local/intel.15
BINDIR          = $(ICCDIR)/bin
# BINDIR        = /rwthfs/rz/SW/UTIL/INTEL/compiler91-07.05.2007/cce/bin
GCC_VERSION	=  

# -----------------------------------------------------------------------------
# Compiler
CC              = $(BINDIR)/icc
LD              = $(BINDIR)/icpc
CXX             = $(BINDIR)/icpc
CXXLD           = $(BINDIR)/icpc

CPPFLAGS        += $(SYS_INCLUDES)

GCC		= /usr/local/gcc-4.8/bin/gcc
GXX             = /usr/local/gcc-4.8/bin/g++
# GCC		= /rwthfs/rz/SW/UTIL.common/gcc/3.4/x86_64-unknown-linux-gnu/bin/gcc

CCFLAGS	= -gcc-name=$(GCC) -gxx-name=$(GXX) -I/usr/include/x86_64-linux-gnu/ -cxxlib		# common for C and C++
# CCFLAGS		= -cxxlib		                # common for C and C++
CXXFLAGS        = $(CCFLAGS)    			# options for C++ compiler
CFLAGS		= $(CCFLAGS)				# options for C compiler

ifndef CXX_MAJOR
CXX_MAJOR       = $(shell $(GCC) --version | head -n 1 | sed -e 's/.*[ \t]\([0-9]\)\.\([0-9]\)\.\([0-9]\)\([ \t].*\)*$$/\1/')
CXX_MINOR       = $(shell $(GCC) --version | head -n 1 | sed -e 's/.*[ \t]\([0-9]\)\.\([0-9]\)\.\([0-9]\)\([ \t].*\)*$$/\2/')
endif

# -----------------------------------------------------------------------------
# compiler options
DEFINES		+= -D_GNU_SOURCE
CCFLAGS		+= -fno-exceptions
CCFLAGS		+= -funsigned-char
CXXFLAGS	+= -fpermissive
CFLAGS		+= -std=c99
CXXFLAGS	+= -std=gnu++0x
CXXFLAGS        +=  -O3 -msse2
# CXXFLAGS        += -xT           # Optimize for Intel Core 2 Duo
# CCFLAGS	+= -Wall
LDFLAGS         += -L$(ICCDIR)/lib/intel64 -lintlc -lsvml
ifdef MODULE_OPENMP
CCFLAGS		+= -openmp
endif
ifdef MODULE_INTEL_MKL
CCFLAGS     += I/opt/intel/mkl/10.2.6.038/include -I/opt/intel/mkl/10.2.6.038/include/fftw -I/opt/intel/mkl/10.2.6.038/include/em64t/lp64 -I/opt/intel/mkl/10.2.6.038/include/32 -I/opt/intel/Compiler/12.1/0.233/mkl/include -I/opt/intel/Compiler/12.1/0.233/mkl/include/fftw -I/opt/intel/Compiler/12.1/0.233/mkl/include/intel64/lp64 -I/opt/intel/Compiler/12.1/0.233/mkl/include/ia32 -L/opt/intel/mkl/10.2.6.038/lib/em64t -L/opt/intel/mkl/10.2.6.038/lib/32 -lrwthmkl -liomp5 -lpthread -L/opt/intel/Compiler/12.1/0.233/mkl/lib/intel64 -L/opt/intel/Compiler/12.1/0.233/mkl/lib/ia32 -lrwthmkl -lpthread
endif

ifeq ($(COMPILE),debug)
CCFLAGS		+= -g
DEFINES		+= -D_GLIBCXX_DEBUG
endif

ifneq ($(COMPILE),release)
# needed to get symbolic function names in stack traces (see Core/Assertions.cc)
LDFLAGS          += -rdynamic
endif

ifeq ($(PROFILE),bprof)
CCFLAGS		+= -g
LDFLAGS		+= /usr/lib/bmon.o
PROF		= bprof
endif
ifeq ($(PROFILE),gprof)
CCFLAGS		+= -pg
LDFLAGS		+= -pg  -static
PROF		= gprof -b
endif
ifeq ($(PROFILE),valgrind)
CCFLAGS		+= -g
LDFLAGS		+=
PROF		= echo "type: valgrind <execuable>"
endif
ifeq ($(PROFILE),purify)
PROFILE		+= "(not supported)"
PROF		=
endif
