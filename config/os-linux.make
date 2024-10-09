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
OPENFST_VERSION = 1.6.3
OPENFSTDIR  = /work/speech/tools/openfst-1.6.3
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
TF_COMPILE_BASE = /work/tools/asr/tensorflow/2.3.4-generic+cuda10.1+mkl/tensorflow

TF_CXXFLAGS  = -fexceptions
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-bin/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-genfiles/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/eigen_archive/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/com_google_protobuf/src/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/com_google_absl/

TF_LDFLAGS  = -L$(TF_COMPILE_BASE)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework
TF_LDFLAGS += -Wl,-rpath -Wl,$(TF_COMPILE_BASE)/bazel-bin/tensorflow

USE_TENSORFLOW_MKL=1
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
LDFLAGS		+= -L/opt/intel/mkl/10.2.6.038/lib/em64t -L/opt/intel/mkl/10.2.6.038/lib/32 -lrwthmkl -liomp5 -lpthread
INCLUDES	+= -I/opt/intel/mkl/10.2.6.038/include -I/opt/intel/mkl/10.2.6.038/include/fftw -I/opt/intel/mkl/10.2.6.038/include/em64t/lp64 -I/opt/intel/mkl/10.2.6.038/include/32
else
ifdef USE_TENSORFLOW_MKL
LDFLAGS   += -L$(TF_COMPILE_BASE)/bazel-tensorflow/external/mkl_linux/lib/
LDFLAGS   += -Wl,-rpath -Wl,$(TF_COMPILE_BASE)/bazel-tensorflow/external/mkl_linux/lib/
INCLUDES  += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/mkl_linux/include/
LDFLAGS   += -lmklml_intel -liomp5
LDFLAGS   += -llapack
else
INCLUDES    += `pkg-config --cflags blas`
INCLUDES    += `pkg-config --cflags lapack`
LDFLAGS     += `pkg-config --libs blas`
LDFLAGS     += `pkg-config --libs lapack`
endif
endif
endif

ifdef MODULE_CUDA
CUDAROOT    = /usr/local/cuda-7.0
INCLUDES    += -I$(CUDAROOT)/include/
LDFLAGS     += -L$(CUDAROOT)/lib64/ -lcublas -lcudart -lcurand
NVCC        = $(CUDAROOT)/bin/nvcc
NVCCFLAGS   = -gencode arch=compute_61,code=sm_61 \  # GTX 1080
	      -gencode arch=compute_75,code=sm_75 \  # RTX 2080
	      -gencode arch=compute_86,code=sm_86 \  # RTX 3090
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
# Use --ldflags --embed for python >= 3.8
PYTHON_PATH = /work/tools/asr/python/3.8.0
ifneq (${PYTHON_PATH},)
INCLUDES    += `${PYTHON_PATH}/bin/python3-config --includes 2>/dev/null`
INCLUDES    += -I${PYTHON_PATH}/lib/python3.8/site-packages/numpy/core/include/
INCLUDES    += -I$(shell python3 -c 'import numpy as np; print(np.get_include())')
INCLUDES    += `python3 -m pybind11 --includes 2>/dev/null`
LDFLAGS     += `${PYTHON_PATH}/bin/python3-config --ldflags --embed 2>/dev/null`
LDFLAGS     += -Wl,-rpath -Wl,${PYTHON_PATH}/lib
else
INCLUDES    += `python3-config --includes 2>/dev/null`
INCLUDES    += -I$(shell python3 -c 'import numpy as np; print(np.get_include())')
INCLUDES    += `python3 -m pybind11 --includes 2>/dev/null`
LDFLAGS     += `python3-config --ldflags --embed 2>/dev/null`
# IF you want to use Python2 for whatever reason:
# INCLUDES    += `pkg-config --cflags python`
# LDFLAGS     += `pkg-config --libs python`
endif
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
ARFLAGS     = rucs
#MAKELIB     = rm -f $@; $(AR) $(ARFLAGS)
ifeq ($(COMPILER),sun)
MAKELIB     = $(CXX) $(CXXFLAGS) -xar -o 
else ifeq ($(COMPILE),debug_dynamic)
MAKELIB     = $(LD) $(LDFLAGS) -shared -o
else
MAKELIB     = $(AR) $(ARFLAGS)
endif
ECHO        = @/bin/echo -e
