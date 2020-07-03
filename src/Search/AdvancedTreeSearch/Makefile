#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintAdvancedTreeSearch.$(a)

LIBSPRINTADVANCEDTREESEARCH_O = $(OBJDIR)/AcousticLookAhead.o \
                        $(OBJDIR)/AdvancedTreeSearch.o \
                        $(OBJDIR)/DynamicBeamPruningStrategy.o \
                        $(OBJDIR)/Helpers.o \
                        $(OBJDIR)/LanguageModelLookahead.o \
                        $(OBJDIR)/PathRecombination.o \
                        $(OBJDIR)/PathRecombinationApproximation.o \
                        $(OBJDIR)/PersistentStateTree.o \
                        $(OBJDIR)/PrefixFilter.o \
                        $(OBJDIR)/ScoreDependentStatistics.o \
                        $(OBJDIR)/SearchSpace.o \
                        $(OBJDIR)/SearchSpaceHelpers.o \
                        $(OBJDIR)/SearchSpaceStatistics.o \
                        $(OBJDIR)/SimpleThreadPool.o \
                        $(OBJDIR)/Trace.o \
                        $(OBJDIR)/TreeBuilder.o \
                        $(OBJDIR)/TreeStructure.o

ifeq ($(OS),darwin)
CCFLAGS += -fexceptions
endif

# These flags make the compilation slow, but are required to inline some critical functions in SearchSpace.cc
CCFLAGS += -Wno-sign-compare -Winline --param max-inline-insns-auto=10000 --param max-inline-insns-single=10000 --param large-function-growth=25000 --param inline-unit-growth=400
CXXFLAGS += -Wno-sign-compare -Winline --param max-inline-insns-auto=10000 --param max-inline-insns-single=10000 --param large-function-growth=25000 --param inline-unit-growth=400

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintAdvancedTreeSearch.$(a): $(LIBSPRINTADVANCEDTREESEARCH_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o check$(exe) $(LDFLAGS)
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTADVANCEDTREESEARCH_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
