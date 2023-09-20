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

# APPTEK SPECIFICS
THIRDPARTY_USR=/opt/thirdparty/usr
INCLUDES += -I$(THIRDPARTY_USR)/include
LDFLAGS += -L$(THIRDPARTY_USR)/lib -Wl,-rpath -Wl,$(THIRDPARTY_USR)/lib

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
TF_CXXFLAGS  = -fexceptions
TF_CXXFLAGS += -I/usr/local/lib/python3.8/dist-packages/tensorflow/include
TF_LDFLAGS  += -Wl,--no-as-needed -Wl,--allow-multiple-definition
TF_LDFLAGS  += -lcrypto
TF_LDFLAGS  += -L/usr/local/lib/tensorflow
TF_LDFLAGS  += -Wl,-rpath -Wl,/usr/local/lib/tensorflow
TF_LDFLAGS  += -ltensorflow_cc -ltensorflow_framework

# USE_TENSORFLOW_MKL=1
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
INCLUDES += -I/usr/include/openblas
LDFLAGS  += -llapack -lopenblas
endif
endif
endif

ifdef MODULE_ONNX
LDFLAGS += -lonnxruntime
ifndef MODULE_TENSORFLOW
CXXFLAGS += -fexceptions
endif
endif


ifdef MODULE_CUDA
CUDAROOT    = /usr/local/cuda-11.6
INCLUDES    += -I$(CUDAROOT)/include/
LDFLAGS     += -L$(CUDAROOT)/lib64/ -lcublas -lcudart -lcurand
NVCC        = $(CUDAROOT)/bin/nvcc
# optimal for GTX680; set sm_35 for K20
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
PYTHON_PATH =
ifneq (${PYTHON_PATH},)
INCLUDES    += `${PYTHON_PATH}/bin/python3-config --includes 2>/dev/null`
LDFLAGS     += `${PYTHON_PATH}/bin/python3-config --ldflags --embed 2>/dev/null`
LDFLAGS     += -Wl,-rpath -Wl,${PYTHON_PATH}/lib
else
INCLUDES    += `python3-config --includes 2>/dev/null`
LDFLAGS     += `python3-config --ldflags --embed 2>/dev/null`
# IF you want to use Python2 for whatever reason:
# INCLUDES    += `pkg-config --cflags python`
# LDFLAGS     += `pkg-config --libs python`
endif
INCLUDES    += -I$(shell python3 -c 'import numpy as np; print(np.get_include())')
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
