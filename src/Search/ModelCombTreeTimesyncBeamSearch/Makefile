#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintModelCombTreeTimesyncBeamSearch.$(a)

LIBSPRINTMODELCOMBTREETIMESYNCBEAMSEARCH_O = $(OBJDIR)/ModelCombTreeTimesyncBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintModelCombTreeTimesyncBeamSearch.$(a): $(LIBSPRINTMODELCOMBTREETIMESYNCBEAMSEARCH_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTMODELCOMBTREETIMESYNCBEAMSEARCH_O:.o=.d)
