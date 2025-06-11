#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintTreeTimesyncBeamSearch.$(a)

LIBSPRINTTREETIMESYNCBEAMSEARCH_O = $(OBJDIR)/TreeTimesyncBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintTreeTimesyncBeamSearch.$(a): $(LIBSPRINTTREETIMESYNCBEAMSEARCH_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTTREETIMESYNCBEAMSEARCH_O:.o=.d)
