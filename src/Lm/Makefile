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
		$(OBJDIR)/CorpusStatistics.o \
		$(OBJDIR)/IndexMap.o \
		$(OBJDIR)/LanguageModel.o \
		$(OBJDIR)/Module.o \
		$(OBJDIR)/NNHistoryManager.o \
		$(OBJDIR)/ReverseArpaLm.o \
		$(OBJDIR)/ScaledLanguageModel.o \
		$(OBJDIR)/WordlistInterface.o


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
ifdef MODULE_LM_FFNN
LIBSPRINTLM_O += $(OBJDIR)/FFNeuralNetworkLanguageModel.o
endif
ifdef MODULE_LM_TFRNN
LIBSPRINTLM_O += $(OBJDIR)/BlasNceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/CompressedVector.o
LIBSPRINTLM_O += $(OBJDIR)/FixedQuantizationCompressedVectorFactory.o
LIBSPRINTLM_O += $(OBJDIR)/LstmStateManager.o
LIBSPRINTLM_O += $(OBJDIR)/NceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/PassthroughSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/QuantizedBlasNceSoftmaxAdapter.o
LIBSPRINTLM_O += $(OBJDIR)/QuantizedCompressedVectorFactory.o
LIBSPRINTLM_O += $(OBJDIR)/ReducedPrecisionCompressedVectorFactory.o
LIBSPRINTLM_O += $(OBJDIR)/TransformerStateManager.o
LIBSPRINTLM_O += $(OBJDIR)/TFRecurrentLanguageModel.o
LIBSPRINTLM_O += $(OBJDIR)/SimpleTransformerLm.o
#MODF DummyCompressedVectorFactory.hh
#MODF SoftmaxAdapter.hh
#MODF StateManager.hh

CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

CHECK_O		= $(OBJDIR)/check.o			\
		  ../Flf/libSprintFlf.$(a) \
		  ../Flf/FlfCore/libSprintFlfCore.$(a) \
		  ../Speech/libSprintSpeech.$(a) \
		  ../Am/libSprintAm.$(a) \
		  ../Mc/libSprintMc.$(a) \
		  ../Bliss/libSprintBliss.$(a) \
		  ../Nn/libSprintNn.$(a) \
		  ../Me/libSprintMe.$(a) \
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
ifdef MODULE_MATH_NR
CHECK_O += ../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_PYTHON
CHECK_O += ../Python/libSprintPython.$(a)
endif
ifdef MODULE_LM_TFRNN
CHECK_O += ../Tensorflow/libSprintTensorflow.$(a)
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
