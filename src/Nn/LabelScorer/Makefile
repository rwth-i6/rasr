#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLabelScorer.$(a)

LIBSPRINTLABELSCORER_O =  \
    $(OBJDIR)/Encoder.o \
    $(OBJDIR)/LabelScorer.o \
    $(OBJDIR)/ScoringContext.o

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLabelScorer.$(a): $(LIBSPRINTLABELSCORER_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLABELSCORER_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
