#!make

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

CCFLAGS		+= -fexceptions
# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= nn-trainer$(exe)

NN_TRAINER_O = $(OBJDIR)/NnTrainer.o \
		../../Am/libSprintAm.$(a) \
		../../Audio/libSprintAudio.$(a) \
		../../Bliss/libSprintBliss.$(a) \
		../../Core/libSprintCore.$(a) \
		../../Flow/libSprintFlow.$(a) \
		../../Fsa/libSprintFsa.$(a) \
		../../Lattice/libSprintLattice.$(a) \
		../../Lm/libSprintLm.$(a) \
		../../Math/Lapack/libSprintMathLapack.$(a) \
		../../Math/libSprintMath.$(a) \
		../../Mc/libSprintMc.$(a) \
		../../Mm/libSprintMm.$(a) \
		../../Nn/libSprintNn.$(a) \
		../../Search/libSprintSearch.$(a) \
		../../Signal/libSprintSignal.$(a) \
		../../Speech/libSprintSpeech.$(a)

ifdef MODULE_CART
NN_TRAINER_O += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_FLF_CORE
NN_TRAINER_O += ../../Flf/FlfCore/libSprintFlfCore.$(a)
endif
ifdef MODULE_MATH_NR
NN_TRAINER_O += ../../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_ADVANCED_TREE_SEARCH
NN_TRAINER_O += ../../Search/AdvancedTreeSearch/libSprintAdvancedTreeSearch.$(a)
endif
ifdef MODULE_SEARCH_WFST
NN_TRAINER_O += ../../Search/Wfst/libSprintSearchWfst.$(a) ../../OpenFst/libSprintOpenFst.$(a)
endif
ifdef MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH
NN_TRAINER_O += ../../Search/GenericSeq2SeqTreeSearch/libSprintGenericSeq2SeqTreeSearch.$(a)
endif
ifdef MODULE_PYTHON
NN_TRAINER_O += ../../Python/libSprintPython.$(a)
endif
ifdef MODULE_TENSORFLOW
NN_TRAINER_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif
ifdef MODULE_ONNX
NN_TRAINER_O += ../../Onnx/libSprintOnnx.$(a)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

nn-trainer$(exe): $(NN_TRAINER_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)
#	$(LD) $^ -o $@ $(LDFLAGS)

nn-trainer$(so): $(NN_TRAINER_O)
	$(LD) -shared $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)


include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(NN_TRAINER_O)))
# DO NOT DELETE
