#!gmake

TOPDIR          = ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS         =
TARGETS         = librasr.so

CXXFLAGS += -fPIC
LDFLAGS += -shared

RASR_LIB_O = $(OBJDIR)/LibRASR.o \
             ../../Flf/libSprintFlf.$(a) \
             ../../Flf/FlfCore/libSprintFlfCore.$(a) \
             ../../Speech/libSprintSpeech.$(a) \
             ../../Lattice/libSprintLattice.$(a) \
             ../../Lm/libSprintLm.$(a) \
             ../../Am/libSprintAm.$(a) \
             ../../Mm/libSprintMm.$(a) \
             ../../Mc/libSprintMc.$(a) \
             ../../Bliss/libSprintBliss.$(a) \
             ../../Audio/libSprintAudio.$(a) \
             ../../Core/libSprintCore.$(a) \
             ../../Fsa/libSprintFsa.$(a) \
             ../../Flow/libSprintFlow.$(a) \
             ../../Signal/libSprintSignal.$(a) \
             ../../Math/libSprintMath.$(a) \
             ../../Math/Lapack/libSprintMathLapack.$(a)

RASR_LIB_O += $(subst src,../..,$(LIBS_SEARCH))
ifdef MODULE_FLF_EXT
RASR_LIB_O += ../../Flf/FlfExt/libSprintFlfExt.$(a)
endif

ifdef MODULE_CART
RASR_LIB_O += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_MATH_NR
RASR_LIB_O += ../../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_PYTHON
RASR_LIB_O += $(OBJDIR)/PybindModule.o \
              ../../Python/libSprintPython.$(a)
CXXFLAGS += $(shell python3 -m pybind11 --includes) -fexceptions
LDFLAGS += $(shell python3-config --ldflags) -Wl,-u,PyInit_librasr
endif
ifdef MODULE_NN
RASR_LIB_O += ../../Nn/libSprintNn.$(a)
endif
ifdef MODULE_ONNX
RASR_LIB_O += ../../Onnx/libSprintOnnx.$(a)
endif
ifdef MODULE_TENSORFLOW
RASR_LIB_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  := $(TF_LDFLAGS) $(LDFLAGS)
endif

CHECK_O = $(OBJDIR)/check.o

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

librasr.so: $(RASR_LIB_O)
	$(LD) $(CXXFLAGS) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

check$(exe): $(CHECK_O) librasr.so
	$(LD) -o $@ $(LDFLAGS) -L. -lrasr

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
