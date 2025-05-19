#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLexiconfreeLabelsyncBeamSearch.$(a)

LIBSPRINTLEXICONFREELABELSYNCBEAMSEARCH_O = $(OBJDIR)/LexiconfreeLabelsyncBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLexiconfreeLabelsyncBeamSearch.$(a): $(LIBSPRINTLEXICONFREELABELSYNCBEAMSEARCH_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLEXICONFREELABELSYNCBEAMSEARCH_O:.o=.d)
