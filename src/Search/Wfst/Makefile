#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintSearchWfst.$(a) check_ctrans$(exe) check$(exe)

LIBSPRINTSEARCHWFST_O = $(OBJDIR)/BookKeeping.o \
                        $(OBJDIR)/Builder.o \
                        $(OBJDIR)/ComposedNetwork.o \
                        $(OBJDIR)/CompressedNetwork.o \
                        $(OBJDIR)/ContextTransducerBuilder.o \
                        $(OBJDIR)/CreateOperations.o \
                        $(OBJDIR)/DynamicLmFst.o \
                        $(OBJDIR)/ExpandingFsaSearch.o \
                        $(OBJDIR)/GrammarFst.o \
                        $(OBJDIR)/IoOperations.o \
                        $(OBJDIR)/Lattice.o \
                        $(OBJDIR)/LatticeAdaptor.o \
                        $(OBJDIR)/LatticeArchive.o \
                        $(OBJDIR)/LatticeGenerator.o \
                        $(OBJDIR)/LatticeHandler.o \
                        $(OBJDIR)/LatticeNetwork.o \
                        $(OBJDIR)/LatticeReader.o \
                        $(OBJDIR)/LexiconBuilder.o \
                        $(OBJDIR)/LexiconFst.o \
                        $(OBJDIR)/Module.o \
                        $(OBJDIR)/NonWordTokens.o \
                        $(OBJDIR)/Network.o \
                        $(OBJDIR)/SearchSpace.o \
                        $(OBJDIR)/StateSequence.o \
                        $(OBJDIR)/StateTree.o \
                        $(OBJDIR)/Traceback.o \
                        $(OBJDIR)/FstOperations.o \
                        $(OBJDIR)/UtilityOperations.o \
                        $(OBJDIR)/WordEnd.o \
                        
#MODF AutomatonAdapter.hh
#MODF ComposeFst.hh
#MODF LabelMapper.hh
#MODF Statistics.hh
#MODF Types.hh

CHECK_CTRANS_O = $(OBJDIR)/check_ctrans.o \
			  ../../Am/libSprintAm.$(a) \
			  ../../Mc/libSprintMc.$(a) \
			  ../../Mm/libSprintMm.$(a) \
			  ../../Cart/libSprintCart.$(a) \
			  ../../Math/libSprintMath.$(a) \
			  ../../Math/Lapack/libSprintMathLapack.$(a) \
			  ../../Math/Nr/libSprintMathNr.$(a) \
			  ../../Legacy/libSprintLegacy.$(a) \
			  ../../Bliss/libSprintBliss.$(a) \
			  ../../Fsa/libSprintFsa.$(a) \
			  ../../Core/libSprintCore.$(a) \
			  ../../OpenFst/libSprintOpenFst.$(a)
			  
CHECK_O = $(OBJDIR)/check.o \
			  libSprintSearchWfst.$(a) \
			  ../libSprintSearch.$(a) \
			  ../../Speech/libSprintSpeech.$(a) \
			  ../../Am/libSprintAm.$(a) \
			  ../../Mm/libSprintMm.$(a) \
			  ../../Legacy/libSprintLegacy.$(a) \
			  ../../Cart/libSprintCart.$(a) \
			  ../../Math/libSprintMath.$(a) \
			  ../../Math/Lapack/libSprintMathLapack.$(a) \
			  ../../Math/Nr/libSprintMathNr.$(a) \
			  ../../Bliss/libSprintBliss.$(a) \
			  ../../Lm/libSprintLm.$(a) \
			  ../../Fsa/libSprintFsa.$(a) \
			  ../../OpenFst/libSprintOpenFst.$(a) \
			  ../../Mc/libSprintMc.$(a) \
			  ../../Core/libSprintCore.$(a)
			  
			  
TARGETS += check_ctrans$(exe)			  

ifeq ($(OS),darwin)
CCFLAGS += -fexceptions
endif
CCFLAGS += -Wno-sign-compare


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintSearchWfst.$(a): $(LIBSPRINTSEARCHWFST_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o check$(exe) $(LDFLAGS)
	
check_ctrans$(exe): $(CHECK_CTRANS_O)
	$(LD) $^ -o $@ $(LDFLAGS)
	
lattice_tropical-arc.so : $(OBJDIR)/Lattice.o $(OBJDIR)/LatticeArc.o ../../Core/libSprintCore.$(a)
	$(CXX) $(LDFLAGS) -shared -Wl,-export-dynamic $^ -o $@
	
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTSEARCHWFST_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
