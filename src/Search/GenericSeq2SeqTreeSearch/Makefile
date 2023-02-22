#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintGenericSeq2SeqTreeSearch.$(a)

LIBSPRINTSEQ2SEQTREESEARCH_O = $(OBJDIR)/Seq2SeqTreeSearch.o \
                               $(OBJDIR)/LabelTree.o \
                               $(OBJDIR)/Seq2SeqSearchSpace.o \
                               $(OBJDIR)/LanguageModelLookahead.o \
                               $(OBJDIR)/Seq2SeqAligner.o

ifeq ($(OS),darwin)
CCFLAGS += -fexceptions
endif

# These flags make the compilation slow, but are required to inline some critical functions in SearchSpace (copied from AdvancedTreeSearch)
CCFLAGS += -Wno-sign-compare -Winline --param max-inline-insns-auto=10000 --param max-inline-insns-single=10000 --param large-function-growth=25000 --param inline-unit-growth=400
CXXFLAGS += -Wno-sign-compare -Winline --param max-inline-insns-auto=10000 --param max-inline-insns-single=10000 --param large-function-growth=25000 --param inline-unit-growth=400

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintGenericSeq2SeqTreeSearch.$(a): $(LIBSPRINTSEQ2SEQTREESEARCH_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o check$(exe) $(LDFLAGS)
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTSEQ2SEQTREESEARCH_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
