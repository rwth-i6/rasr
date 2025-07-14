#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 		=
TARGETS			= lm-util$(exe)

LM_UTIL_TOOL_O = $(OBJDIR)/LmUtilityTool.o \
		 ../../Flf/libSprintFlf.$(a) \
		 ../../Flf/FlfCore/libSprintFlfCore.$(a) \
		 ../../Speech/libSprintSpeech.$(a) \
		 ../../Am/libSprintAm.$(a) \
		 ../../Mc/libSprintMc.$(a) \
		 ../../Bliss/libSprintBliss.$(a) \
		 ../../Nn/libSprintNn.$(a) \
		 ../../Nn/LabelScorer/libSprintLabelScorer.$(a) \
		 ../../Mm/libSprintMm.$(a) \
		 ../../Signal/libSprintSignal.$(a) \
		 ../../Flow/libSprintFlow.$(a) \
		 ../../Math/libSprintMath.$(a) \
		 ../../Math/Lapack/libSprintMathLapack.$(a) \
		 ../../Search/libSprintSearch.$(a) \
		 $(subst src,../..,$(LIBS_SEARCH)) \
		 ../../Lattice/libSprintLattice.$(a) \
		 ../../Fsa/libSprintFsa.$(a) \
		 ../../Lm/libSprintLm.$(a) \
		 ../../Core/libSprintCore.$(a)

ifdef MODULE_FLF_EXT
LM_UTIL_TOOL_O   += ../../Flf/FlfExt/libSprintFlfExt.$(a)
endif

ifdef MODULE_CART
LM_UTIL_TOOL_O   += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_MATH_NR
LM_UTIL_TOOL_O   += ../../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_SPARSE
LM_UTIL_TOOL_O   += ../../Sparse/libSprintSparse.$(a)
endif
ifdef MODULE_ONNX
LM_UTIL_TOOL_O   += ../../Onnx/libSprintOnnx.$(a)
endif
ifdef MODULE_PYTHON
LM_UTIL_TOOL_O   += ../../Python/libSprintPython.$(a)
endif
ifdef MODULE_ONNX
LM_UTIL_TOOL_O += ../../Onnx/libSprintOnnx.$(a)
endif
ifdef MODULE_TENSORFLOW
LM_UTIL_TOOL_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

lm-util$(exe): $(LM_UTIL_TOOL_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(LM_UTIL_TOOL_O)))
