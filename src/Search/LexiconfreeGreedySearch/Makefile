#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLexiconfreeGreedySearch.$(a)

LIBSPRINTLEXICONFREEGREEDYSEARCH_O = $(OBJDIR)/LexiconfreeGreedySearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLexiconfreeGreedySearch.$(a): $(LIBSPRINTLEXICONFREEGREEDYSEARCH_O)
	$(MAKELIB) $@ $^
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLEXICONFREEGREEDYSEARCH_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
