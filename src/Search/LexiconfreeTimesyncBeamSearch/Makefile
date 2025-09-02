#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLexiconfreeTimesyncBeamSearch.$(a)

LIBSPRINTLEXICONFREETIMESYNCBEAMSEARCH_O = $(OBJDIR)/LexiconfreeTimesyncBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLexiconfreeTimesyncBeamSearch.$(a): $(LIBSPRINTLEXICONFREETIMESYNCBEAMSEARCH_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLEXICONFREETIMESYNCBEAMSEARCH_O:.o=.d)
