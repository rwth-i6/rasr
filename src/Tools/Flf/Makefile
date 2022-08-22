#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= flf-tool$(exe)

FLF_TOOL_O	= $(OBJDIR)/FlfTool.o \
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
		  ../../Fsa/libSprintFsa.$(a) 

FLF_TOOL_O += $(subst src,../..,$(LIBS_SEARCH))

ifdef MODULE_FLF_EXT
FLF_TOOL_O += ../../Flf/FlfExt/libSprintFlfExt.$(a)
endif

FLF_TOOL_O += ../../Signal/libSprintSignal.$(a)
FLF_TOOL_O += ../../Flow/libSprintFlow.$(a)
FLF_TOOL_O += ../../Math/libSprintMath.$(a)
FLF_TOOL_O += ../../Math/Lapack/libSprintMathLapack.$(a)

ifdef MODULE_CART
FLF_TOOL_O += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_MATH_NR
FLF_TOOL_O += ../../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_PYTHON
FLF_TOOL_O += ../../Python/libSprintPython.$(a)
endif
ifdef MODULE_NN
FLF_TOOL_O += ../../Nn/libSprintNn.$(a)
endif
ifdef MODULE_TENSORFLOW
FLF_TOOL_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

flf-tool$(exe): $(FLF_TOOL_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(FLF_TOOL_O)))
