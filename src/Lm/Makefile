#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	= 
TARGETS		= libSprintLm.$(a) check$(exe)

LIBSPRINTLM_O	= \
		$(OBJDIR)/AbstractNNLanguageModel.o \
		$(OBJDIR)/BackingOff.o \
		$(OBJDIR)/ClassLm.o \
		$(OBJDIR)/CombineLm.o \
		$(OBJDIR)/Compose.o \
		$(OBJDIR)/CompressedVector.o \
		$(OBJDIR)/CorpusStatistics.o \
		$(OBJDIR)/FixedQuantizationCompressedVectorFactory.o \
		$(OBJDIR)/IndexMap.o \
		$(OBJDIR)/LanguageModel.o \
		$(OBJDIR)/Module.o \
		$(OBJDIR)/NNHistoryManager.o \
		$(OBJDIR)/QuantizedCompressedVectorFactory.o \
		$(OBJDIR)/RecurrentLanguageModel.o \
		$(OBJDIR)/ReducedPrecisionCompressedVectorFactory.o \
		$(OBJDIR)/ReverseArpaLm.o \
		$(OBJDIR)/ScaledLanguageModel.o \
		$(OBJDIR)/WordlistInterface.o

#MODF AbstractStateManager.hh
#MODF DummyCompressedVectorFactory.hh

ifdef MODULE_LM_ARPA
LIBSPRINTLM_O += $(OBJDIR)/ArpaLm.o
endif

ifdef MODULE_LM_FSA
LIBSPRINTLM_O += $(OBJDIR)/FsaLm.o
LIBSPRINTLM_O += $(OBJDIR)/CheatingSegmentLm.o
endif

ifdef MODULE_LM_ZEROGRAM
LIBSPRINTLM_O += $(OBJDIR)/Zerogram.o
endif
ifdef MODULE_LM_TFRNN
LIBSPRINTLM_O += $(OBJDIR)/TFBlasNceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/TFLstmStateManager.o
LIBSPRINTLM_O += $(OBJDIR)/TFNceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/TFPassthroughSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/TFQuantizedBlasNceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/TFRecurrentLanguageModel.o
LIBSPRINTLM_O += $(OBJDIR)/TFTransformerStateManager.o
#MODF TFSoftmaxAdapter.hh

CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

ifdef MODULE_LM_ONNX
LIBSPRINTLM_O += $(OBJDIR)/OnnxLstmStateManager.o
LIBSPRINTLM_O += $(OBJDIR)/OnnxPassthroughSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/OnnxNceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/OnnxRecurrentLanguageModel.o
#MODF OnnxStateVariable.hh
#MODF OnnxSoftmaxAdapter.hh
endif

CHECK_O		= $(OBJDIR)/check.o			\
		  ../Flf/libSprintFlf.$(a) \
		  ../Flf/FlfCore/libSprintFlfCore.$(a) \
		  ../Speech/libSprintSpeech.$(a) \
		  ../Am/libSprintAm.$(a) \
		  ../Mc/libSprintMc.$(a) \
		  ../Bliss/libSprintBliss.$(a) \
		  ../Nn/libSprintNn.$(a) \
		  ../Mm/libSprintMm.$(a) \
		  ../Signal/libSprintSignal.$(a) \
		  ../Flow/libSprintFlow.$(a) \
		  ../Math/libSprintMath.$(a) \
		  ../Math/Lapack/libSprintMathLapack.$(a)

CHECK_O += $(subst src,..,$(LIBS_SEARCH))
CHECK_O += ../Lattice/libSprintLattice.$(a)
CHECK_O += ../Fsa/libSprintFsa.$(a) 
CHECK_O += libSprintLm.$(a)
CHECK_O += ../Core/libSprintCore.$(a)

ifdef MODULE_CART
CHECK_O += ../Cart/libSprintCart.$(a)
endif
ifdef MODULE_FLF_EXT
CHECK_O += ../Flf/FlfExt/libSprintFlfExt.$(a)
endif
ifdef MODULE_MATH_NR
CHECK_O += ../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_PYTHON
CHECK_O += ../Python/libSprintPython.$(a)
endif
ifdef MODULE_FLF_EXT
CHECK_O 	+= ../Flf/FlfExt/libSprintFlfExt.$(a)
endif
ifdef MODULE_LM_TFRNN
CHECK_O += ../Tensorflow/libSprintTensorflow.$(a)
endif
ifdef MODULE_LM_ONNX
CHECK_O += ../Onnx/libSprintOnnx.$(a)
endif

# -----------------------------------------------------------------------------

all: $(TARGETS)

RwthLm:
	$(MAKE) -C $@

.PHONY:	$(SUBDIRS)

libSprintLm.$(a): $(SUBDIRS) $(LIBSPRINTLM_O) 
	$(MAKELIB) $@ $(LIBSPRINTLM_O) $(ADD_FILES)

check$(exe): $(CHECK_O)
	$(LD) $^ -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

CXXFLAGS += -fexceptions

sinclude $(LIBSPRINTLM_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
# DO NOT DELETE

BackingOff.o: BackingOff.hh
# BackingOff.hh includes:
#	LanguageModel.hh
BackingOff.o: LanguageModel.hh
