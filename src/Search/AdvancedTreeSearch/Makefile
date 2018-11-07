#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintAdvancedTreeSearch.$(a)

LIBSPRINTADVANCEDTREESEARCH_O = $(OBJDIR)/AdvancedTreeSearch.o \
                        $(OBJDIR)/PersistentStateTree.o \
                        $(OBJDIR)/ScoreDependentStatistics.o \
                        $(OBJDIR)/SearchSpace.o \
                        $(OBJDIR)/SearchSpaceStatistics.o \
                        $(OBJDIR)/Trace.o \
                        $(OBJDIR)/TraceManager.o \
                        $(OBJDIR)/SearchSpaceHelpers.o \
                        $(OBJDIR)/TreeStructure.o \
                        $(OBJDIR)/LanguageModelLookahead.o \
                        $(OBJDIR)/AcousticLookAhead.o \
                        $(OBJDIR)/PathRecombination.o \
                        $(OBJDIR)/PathRecombinationApproximation.o \
                        $(OBJDIR)/Helpers.o \
                        $(OBJDIR)/SimpleThreadPool.o \
                        $(OBJDIR)/TreeBuilder.o \
                        $(OBJDIR)/PrefixFilter.o

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
