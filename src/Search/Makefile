#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintSearch.$(a) check$(exe)

LIBSPRINTSEARCH_O	= \
		$(OBJDIR)/Aligner.o \
		$(OBJDIR)/LatticeHandler.o \
		$(OBJDIR)/LanguageModelLookahead.o \
		$(OBJDIR)/Module.o \
		$(OBJDIR)/Search.o \
		$(OBJDIR)/StateTree.o \
		$(OBJDIR)/StateTreeIo.o \
		$(OBJDIR)/Traceback.o \
		$(OBJDIR)/WordConditionedTreeSearch.o

CHECK_O		= $(OBJDIR)/check.o \
			  libSprintSearch.$(a) \
			  ../Bliss/libSprintBliss.$(a) \
			  ../Fsa/libSprintFsa.$(a) \
			  ../Core/libSprintCore.$(a)
			  

ifdef MODULE_SEARCH_MBR
LIBSPRINTSEARCH_O += $(OBJDIR)/MinimumBayesRiskSearch.o
LIBSPRINTSEARCH_O += $(OBJDIR)/MinimumBayesRiskAStarSearch.o
LIBSPRINTSEARCH_O += $(OBJDIR)/MinimumBayesRiskNBestListSearch.o
LIBSPRINTSEARCH_O += $(OBJDIR)/MinimumBayesRiskSearchUtil.o
endif
ifdef MODULE_SEARCH_WFST
SUBDIRS += Wfst
endif
ifdef MODULE_ADVANCED_TREE_SEARCH
SUBDIRS += AdvancedTreeSearch
endif
ifdef MODULE_SEARCH_LINEAR
LIBSPRINTSEARCH_O += $(OBJDIR)/LinearSearch.o
endif
SUBDIRS += LexiconfreeBeamSearch
SUBDIRS += LexiconfreeGreedySearch


# -----------------------------------------------------------------------------

build: libSprintSearch.$(a) $(SUBDIRS)

all: $(TARGETS)

.PHONY: $(SUBDIRS)

libSprintSearch.$(a): $(LIBSPRINTSEARCH_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o check$(exe) $(LDFLAGS)

Wfst: 
	$(MAKE) -C $@ libSprintSearchWfst.$(a)

AdvancedTreeSearch:
	$(MAKE) -C $@ libSprintAdvancedTreeSearch.$(a)

LexiconfreeBeamSearch:
	$(MAKE) -C $@ libSprintLexiconfreeBeamSearch.$(a)

LexiconfreeGreedySearch:
	$(MAKE) -C $@ libSprintLexiconfreeGreedySearch.$(a)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTSEARCH_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))

# LanguageModelLookahead.cc causes an internal compiler error
# when using icpc -> compile it using g++
ifeq ($(COMPILER),icc)
$(OBJDIR)/LanguageModelLookahead.o : 
	make COMPILER=gcc $@
endif
