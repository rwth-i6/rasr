#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintUnconstrainedBeamSearch.$(a)

LIBSPRINTUNCONSTRAINEDBEAMSEARCH_O = $(OBJDIR)/UnconstrainedBeamSearch.o


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintUnconstrainedBeamSearch.$(a): $(LIBSPRINTUNCONSTRAINEDBEAMSEARCH_O)
	$(MAKELIB) $@ $^
	
include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTUNCONSTRAINEDBEAMSEARCH_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
