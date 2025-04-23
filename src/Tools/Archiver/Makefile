#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= archiver$(exe)

ARCHIVER_O	= $(OBJDIR)/Archiver.o \
                 ../../Speech/libSprintSpeech.$(a) \
                 $(subst src,../..,$(LIBS_SEARCH)) \
                 ../../Lattice/libSprintLattice.$(a) \
                 ../../Lm/libSprintLm.$(a) \
                 ../../Flf/libSprintFlf.$(a) \
                 ../../Flf/FlfCore/libSprintFlfCore.$(a) \
                 ../../Mc/libSprintMc.$(a) \
                 ../../Am/libSprintAm.$(a) \
                 ../../Mm/libSprintMm.$(a) \
                 ../../Cart/libSprintCart.$(a) \
                 ../../Signal/libSprintSignal.$(a) \
                 ../../Bliss/libSprintBliss.$(a) \
                 ../../Math/libSprintMath.$(a) \
                 ../../Math/Lapack/libSprintMathLapack.$(a) \
                 ../../Math/Nr/libSprintMathNr.$(a) \
                 ../../Core/libSprintCore.$(a) \
                 ../../Flow/libSprintFlow.$(a) \
                 ../../Fsa/libSprintFsa.$(a)

ifdef MODULE_PYTHON
ARCHIVER_O += ../../Python/libSprintPython.$(a)
endif
ifdef MODULE_NN
ARCHIVER_O += ../../Nn/libSprintNn.$(a)
endif
ifdef MODULE_OPENFST
ARCHIVER_O += ../../OpenFst/libSprintOpenFst.$(a)
endif
ifdef MODULE_ONNX
ARCHIVER_O += ../../Onnx/libSprintOnnx.$(a)
endif
ifdef MODULE_LM_TFRNN
ARCHIVER_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

archiver$(exe): $(ARCHIVER_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(ARCHIVER_O)))
