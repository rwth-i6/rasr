#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintUnconstrainedGreedySearch.$(a)

LIBSPRINTUNCONSTRAINEDGREEDYSEARCH_O = $(OBJDIR)/UnconstrainedGreedySearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintUnconstrainedGreedySearch.$(a): $(LIBSPRINTUNCONSTRAINEDGREEDYSEARCH_O)
	$(MAKELIB) $@ $^
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTUNCONSTRAINEDGREEDYSEARCH_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
