#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLexiconfreeBeamSearch.$(a)

LIBSPRINTLEXICONFREEBEAMSEARCH_O = $(OBJDIR)/LexiconfreeBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLexiconfreeBeamSearch.$(a): $(LIBSPRINTLEXICONFREEBEAMSEARCH_O)
	$(MAKELIB) $@ $^
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLEXICONFREEBEAMSEARCH_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
