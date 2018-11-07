#!make

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= speech-recognizer$(exe)

COMMON_O	= ../../Speech/libSprintSpeech.$(a) \
		  ../../Lattice/libSprintLattice.$(a) \
		  ../../Am/libSprintAm.$(a) \
		  ../../Lm/libSprintLm.$(a) \
		  ../../Mm/libSprintMm.$(a) \
		  ../../Mc/libSprintMc.$(a) \
		  ../../Bliss/libSprintBliss.$(a) \
		  ../../Audio/libSprintAudio.$(a) \
		  ../../Signal/libSprintSignal.$(a) \
		  ../../Flow/libSprintFlow.$(a) \
		  ../../Math/libSprintMath.$(a) \
		  ../../Math/Lapack/libSprintMathLapack.$(a) \
		  ../../Core/libSprintCore.$(a) \
		  ../../Fsa/libSprintFsa.$(a)

COMMON_O += $(subst src,../..,$(LIBS_SEARCH))

ifdef MODULE_CART
COMMON_O += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_FLF_CORE
COMMON_O += ../../Flf/FlfCore/libSprintFlfCore.$(a)
endif
ifdef MODULE_OPENFST
ifeq ($(OS),darwin)
CCFLAGS += -fexceptions
endif
endif
ifdef MODULE_MATH_NR
COMMON_O +=	../../Math/Nr/libSprintMathNr.$(a)
endif
ifdef MODULE_SEARCH_WFST
COMMON_O += ../../Search/Wfst/libSprintSearchWfst.$(a)
TARGETS += fsa-search-builder$(exe)
endif
ifdef MODULE_PYTHON
COMMON_O += ../../Python/libSprintPython.$(a)
endif
ifdef MODULE_NN
COMMON_O += ../../Nn/libSprintNn.$(a)
endif
ifdef MODULE_TENSORFLOW
COMMON_O += ../../Tensorflow/libSprintTensorflow.$(a)
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

# -----------------------------------------------------------------------------

SPEECH_RECOGNIZER_O = $(OBJDIR)/SpeechRecognizer.o \
		      $(COMMON_O)

FSA_SEARCH_BUILDER_O = $(OBJDIR)/FsaSearchBuilder.o \
			  $(COMMON_O)		   

$(OBJDIR)/FsaSearchBuilder.o : CCFLAGS += -Wno-sign-compare

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

speech-recognizer$(exe):  $(SPEECH_RECOGNIZER_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

fsa-search-builder$(exe):  $(FSA_SEARCH_BUILDER_O)
	$(LD) $(LD_START_GROUP) $^ $(LD_END_GROUP) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(SPEECH_RECOGNIZER_O)))
ifdef MODULE_SEARCH_WFST
sinclude $(patsubst %.o,%.d,$(filter %.o,$(FSA_SEARCH_BUILDER_O)))
endif
