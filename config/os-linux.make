#!gmake

LD_START_GROUP = -Wl,-\(
LD_END_GROUP   = -Wl,-\)

# -----------------------------------------------------------------------------
# GNU bison

BISON           = bison
BISON_MAJOR = $(shell $(BISON) --version | grep bison | sed -e 's/.* \([0-9]\+\)\.\([0-9]\+\).*/\1/')
BISON_MINOR = $(shell $(BISON) --version | grep bison | sed -e 's/.* \([0-9]\+\)\.\([0-9]\+\).*/\2/')
CXXFLAGS        += -DBISON_VERSION_$(BISON_MAJOR)

# -----------------------------------------------------------------------------
# protobuf

PROTOC = protoc

# -----------------------------------------------------------------------------
# special libraries
#
# If the libraries are not installed in a standard path, you need to setup the
# PKG_CONFIG_PATH variable. For example (assuming you are using a sh-based
# shell, like bash or zsh, and have installed/configured using './configure --prefix=$HOME')
#
#   PKG_CONFIG_PATH=${HOME}/lib/pkgconfig:${PKG_CONFIG_PATH}
#   export PKG_CONFIG_PATH
#

ifdef MODULE_AUDIO_FFMPEG
INCLUDES    += `pkg-config --cflags libavformat libswresample libavcodec libavutil`
LDFLAGS     += `pkg-config --libs   libavformat libswresample libavcodec libavutil`
endif

ifdef MODULE_OPENFST
_ADD_OPENFST=1
endif

ifdef _ADD_OPENFST
OPENFST_VERSION = 1.6.5
OPENFSTDIR  = /opt/openfst
OPENFSTLIBS = -lfst
INCLUDES    += -isystem $(OPENFSTDIR)/include
DEFINES     += -DOPENFST_$(shell echo $(OPENFST_VERSION) | tr . _)
endif

ifdef MODULE_TBB
TBB_DIR     = /usr/local/intel/tbb/4.0
INCLUDES    += -I$(TBB_DIR)/include
LDFLAGS     += -L$(TBB_DIR)/lib -ltbb
endif

ifdef MODULE_TENSORFLOW
ifndef TF_INCLUDEDIR
  TF_INCLUDEDIR := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
endif

ifndef TF_LIBDIR
  TF_LIBDIR := "/opt/apptek/thirdparty/usr/lib/"
endif

TF_CXXFLAGS := -fexceptions
TF_CXXFLAGS += -I$(TF_INCLUDEDIR)
TF_LDFLAGS  := -L$(TF_LIBDIR)
TF_LDFLAGS  += -liomp5 -ltensorflow_cc -ltensorflow_framework
endif

ifdef MODULE_TEST
#LDFLAGS += -L/usr/lib/x86_64-linux-gnu
#INCLUDES = -I/usr/nonstandard/path/include
endif

# -----------------------------------------------------------------------------
# system Libraries

# PThreads
LDFLAGS     += -lpthread
# Required for profiling, eg. clock_gettime
LDFLAGS     += -lrt

# Common C++
#INCLUDES   += -I$(shell ccgnu2-config --includes)
#LDFLAGS    +=   $(shell ccgnu2-config --stdlibs)
#DEFINES    += -DCCGNU_VERSION=$(shell ccgnu2-config --version | cut --delimiter '.' --fields 2)

# libXML
INCLUDES    += $(shell xml2-config --cflags)
LDFLAGS     += $(shell xml2-config --libs)

# zlib
LDFLAGS     += -lz

# Lapack and Blas
ifdef MODULE_ACML
LDFLAGS     += -L/usr/local/acml-4.4.0/gfortran64_mp/lib/ -lacml_mp -lacml_mv
LDFLAGS     += -L/usr/local/acml-4.4.0/cblas_mp/lib -lcblas_acml
INCLUDES    += -I/usr/local/acml-4.4.0/gfortran64/include/
LDFLAGS     += -llapack 
else
ifdef MODULE_INTEL_MKL
MKL_PATH    = /opt/apptek/src/intel/compilers_and_libraries_2019.2.187
LDFLAGS     += -L$(MKL_PATH)/linux/compiler/lib/intel64_lin -liomp5 
LDFLAGS     += -L$(MKL_PATH)/linux/mkl/lib/intel64_lin -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
INCLUDES    += -I$(MKL_PATH)/linux/mkl/include/
else
INCLUDES    += -I/usr/include/openblas
LDFLAGS     += -llapack -lopenblas
#INCLUDES    += `pkg-config --cflags blas`
#INCLUDES    += `pkg-config --cflags lapack`
#LDFLAGS     += `pkg-config --libs blas`
#LDFLAGS     += `pkg-config --libs lapack`
endif
endif

ifdef MODULE_CUDA
CUDAROOT    = /usr/local/cuda
INCLUDES    += -I$(CUDAROOT)/include/
LDFLAGS     += -L$(CUDAROOT)/lib64/ -lcublas -lcudart -lcurand
NVCC        = $(CUDAROOT)/bin/nvcc
# optimal for GTX680; set sm_35 for K20
NVCCFLAGS   = -gencode arch=compute_20,code=sm_20 \
              -gencode arch=compute_30,code=sm_30 \
	      -gencode arch=compute_35,code=sm_35 \
	      -gencode arch=compute_52,code=sm_52 \
	      -gencode arch=compute_61,code=sm_61 \
              --compiler-options -fPIC
endif

ifeq ($(PROFILE),gprof)
# This works around a problem with the current installation of Lapack at i6
LDFLAGS     += -Xlinker --allow-multiple-definition
endif

# Free Lossless Audio Codec
ifdef MODULE_AUDIO_FLAC
LDFLAGS     += -lFLAC
endif
ifdef MODULE_AUDIO_WAV_SYSTEM
LDFLAGS     += -lsndfile
endif

# C library
ifeq ($(PROFILE),gprof)
LDFLAGS     += -lm_p
LDFLAGS     += -ldl_p -lpthread_p
LDFLAGS     += -lc_p
else
LDFLAGS     += -lm
endif

ifdef MODULE_PYTHON
ifneq ($(shell which python3-config 2>/dev/null),)
PYTHON_CONFIG = python3-config
else
# in CentOS 7, the tool is called python-config, even if it comes with Python 3;
# make sure it returns paths to Python 3 location
PYTHON_CONFIG = python-config
endif
INCLUDES    += -I$(shell python3 -c 'import numpy as np; print(np.get_include())')
INCLUDES    += $(shell $(PYTHON_CONFIG) --includes 2>/dev/null || pkg-config --cflags python3)
LDFLAGS     += $(shell $(PYTHON_CONFIG) --libs 2>/dev/null || pkg-config --libs python3)
LDFLAGS     += -lpython3.8
endif

#INCLUDES    += -I/library/data4/tools/boost_1_66_0
#LDFLAGS     += -L/library/data4/tools/boost_1_66_0/stage/lib

ifdef MODULE_FLOW_REMOTE
LDFLAGS     += -L$(APPTEK_THIRDPARTY_USR)/lib
LDFLAGS     += -lboost_system
endif

# X11 and QT
X11_INCLUDE = -I/usr/X11R6/include
X11_LIB     = -L/usr/X11R6/lib
QT_INCLUDE  = -I/usr/include/qt3
QT_LIB      = -L/usr/lib -lqt
MOC         = moc
XLDFLAGS    = $(X11_LIB) -lXpm -lXext -lX11

# -----------------------------------------------------------------------------
MAKE        = make
MAKEDEPEND  = makedepend -v -D__GNUC__=3 -D__GNUC_MINOR__=3
# AR usage: either 'ar rucs' or 'rm -f $@; ar qcs'
AR          = ar
ARFLAGS     = qcs
ifeq ($(COMPILER),sun)
MAKELIB     = $(CXX) $(CXXFLAGS) -xar -o 
else ifeq ($(COMPILE),debug_dynamic)
MAKELIB     = $(LD) $(LDFLAGS) -shared -o
else
MAKELIB     = $(AR) $(ARFLAGS)
endif
#MAKELIB     = rm -f $@; $(AR) qcs
ECHO        = @/bin/echo -e
