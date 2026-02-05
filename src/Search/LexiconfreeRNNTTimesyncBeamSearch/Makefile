#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLexiconfreeRNNTTimesyncBeamSearch.$(a)

LIBSPRINTLEXICONFREERNNTTIMESYNCBEAMSEARCH_O = $(OBJDIR)/LexiconfreeRNNTTimesyncBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLexiconfreeRNNTTimesyncBeamSearch.$(a): $(LIBSPRINTLEXICONFREERNNTTIMESYNCBEAMSEARCH_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLEXICONFREERNNTTIMESYNCBEAMSEARCH_O:.o=.d)
