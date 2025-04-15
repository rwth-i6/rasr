#!make

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= costa$(exe)

CORPUS_STATISTICS_O = $(OBJDIR)/Costa.o \
		  ../../Lm/libSprintLm.$(a) \
		  ../../Mc/libSprintMc.$(a) \
		  ../../Bliss/libSprintBliss.$(a) \
		  ../../Fsa/libSprintFsa.$(a) \
		  ../../Flf/FlfCore/libSprintFlfCore.$(a) \
		  ../../Audio/libSprintAudio.$(a) \
		  ../../Signal/libSprintSignal.$(a) \
		  ../../Speech/libSprintSpeech.$(a) \
		  ../../Am/libSprintAm.$(a) \
		  ../../Mm/libSprintMm.$(a) \
		  ../../Flow/libSprintFlow.$(a) \
		  ../../Core/libSprintCore.$(a) \
		  ../../Math/libSprintMath.$(a) \
		  ../../Math/Lapack/libSprintMathLapack.$(a) \
		  ../../Lattice/libSprintLattice.$(a)

CORPUS_STATISTICS_O += $(subst src,../..,$(LIBS_SEARCH))

ifdef MODULE_CART
CORPUS_STATISTICS_O += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_ONNX
CORPUS_STATISTICS_O += ../../Onnx/libSprintOnnx.$(a)
endif
ifdef MODULE_MATH_NR
CORPUS_STATISTICS_O += ../../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_PYTHON
CORPUS_STATISTICS_O += ../../Python/libSprintPython.$(a)
endif
ifdef MODULE_NN
CORPUS_STATISTICS_O += ../../Nn/libSprintNn.$(a)
CORPUS_STATISTICS_O += ../../Nn/LabelScorer/libSprintLabelScorer.$(a)
endif
ifdef MODULE_ONNX
CORPUS_STATISTICS_O += ../../Onnx/libSprintOnnx.$(a)
endif
ifdef MODULE_TENSORFLOW
CORPUS_STATISTICS_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

costa$(exe): $(CORPUS_STATISTICS_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(CORPUS_STATISTICS_O)))
