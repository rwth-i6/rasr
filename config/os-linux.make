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

# FFmpeg-SVN
ifdef MODULE_FFMPEG
INCLUDES    += `pkg-config --cflags libavcodec --cflags libavformat --cflags libavutil --cflags libswscale`
LDFLAGS     += `pkg-config --libs libavcodec --libs libavformat --libs libavutil --libs libswscale`
DEFINES     += -DWITH_FFMPEG
#ifeq ($(shell pkg-config --cflags libavutil --atleast-version=50 && echo 1),1)
ifeq ($(shell pkg-config --cflags libavformat --atleast-version=52 && echo 1),1)
DEFINES     += -DFFMPEG_VERSION=50 -D__STDC_CONSTANT_MACROS
LDFLAGS     += $(shell pkg-config --libs libavdevice)
else
DEFINES     += -DFFMPEG_VERSION=49
endif
endif

ifdef MODULE_AUDIO_FFMPEG
INCLUDES    += `pkg-config --cflags libavformat libswresample libavcodec libavutil`
LDFLAGS     += `pkg-config --libs   libavformat libswresample libavcodec libavutil`
endif

# OpenCV
ifdef MODULE_OPENCV
INCLUDES    += `pkg-config --cflags opencv`
LDFLAGS     += `pkg-config --libs opencv` -lrt
DEFINES     += -DWITH_OPENCV
endif

ifdef MODULE_MM_NN
INCLUDES    += `pkg-config --cflags libkdtree`
LDFLAGS     += `pkg-config --libs libkdtree`
endif

ifdef MODULE_OPENFST
_ADD_OPENFST=1
endif
ifdef MODULE_KWS
_ADD_OPENFST=1
endif

ifdef _ADD_OPENFST
OPENFST_VERSION = 1.6.3
OPENFSTDIR  = /work/speech/tools/openfst-$(OPENFST_VERSION)
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
TF_COMPILE_BASE = /work/tools/asr/tensorflow/1.15-generic+cuda10.1/tensorflow

TF_CXXFLAGS  = -fexceptions
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-genfiles/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/eigen_archive/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/com_google_protobuf/src/
TF_CXXFLAGS += -I$(TF_COMPILE_BASE)/bazel-tensorflow/external/com_google_absl/

TF_LDFLAGS  = -L$(TF_COMPILE_BASE)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework
TF_LDFLAGS += -Wl,-rpath -Wl,$(TF_COMPILE_BASE)/bazel-bin/tensorflow
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
INCLUDES    += `pkg-config --cflags blas`
INCLUDES    += `pkg-config --cflags lapack`
LDFLAGS     += `pkg-config --libs blas`
LDFLAGS     += `pkg-config --libs lapack`
endif
endif

ifdef MODULE_CUDA
CUDAROOT    = /usr/local/cuda-7.0
INCLUDES    += -I$(CUDAROOT)/include/
LDFLAGS     += -L$(CUDAROOT)/lib64/ -lcublas -lcudart -lcurand
NVCC        = $(CUDAROOT)/bin/nvcc
# optimal for GTX680; set sm_35 for K20
NVCCFLAGS   = -gencode arch=compute_20,code=sm_20 \
	      -gencode arch=compute_30,code=sm_30 \
	      -gencode arch=compute_35,code=sm_35 \
	      -gencode arch=compute_52,code=sm_52 \
	      -gencode arch=compute_61,code=sm_61
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

ifdef MODULE_IMAGE
# Image libraries like PNG, JPEG, PAM, ...
LDFLAGS     += -lpng -ljpeg -lnetpbm
# FFT
endif
ifdef MODULE_IMAGE_FFTW
LDFLAGS     += -lfftw
endif

ifdef MODULE_PYTHON
INCLUDES    += `/work/tools/asr/python/3.8.0/bin/python3-config --includes 2>/dev/null || pkg-config --cflags python`
LDFLAGS     += `/work/tools/asr/python/3.8.0/bin/python3-config --ldflags 2>/dev/null || pkg-config --ldflags python` -lpython3.8 -Wl,-rpath -Wl,/work/tools/asr/python/3.8.0/lib
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
