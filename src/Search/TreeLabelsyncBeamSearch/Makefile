#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintTreeLabelsyncBeamSearch.$(a)

LIBSPRINTTREELABELSYNCBEAMSEARCH_O = $(OBJDIR)/TreeLabelsyncBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintTreeLabelsyncBeamSearch.$(a): $(LIBSPRINTTREELABELSYNCBEAMSEARCH_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTTREELABELSYNCBEAMSEARCH_O:.o=.d)
