#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLabelScorer.$(a)

LIBSPRINTLABELSCORER_O =  \
    $(OBJDIR)/BufferedLabelScorer.o \
    $(OBJDIR)/CtcPrefixLabelScorer.o \
    $(OBJDIR)/CombineLabelScorer.o \
    $(OBJDIR)/DataView.o \
    $(OBJDIR)/Encoder.o \
    $(OBJDIR)/EncoderDecoderLabelScorer.o \
    $(OBJDIR)/EncoderFactory.o \
    $(OBJDIR)/FixedContextOnnxLabelScorer.o \
    $(OBJDIR)/LabelScorer.o \
    $(OBJDIR)/LabelScorerFactory.o \
    $(OBJDIR)/NoContextOnnxLabelScorer.o \
    $(OBJDIR)/NoOpLabelScorer.o \
    $(OBJDIR)/PriorLabelScorer.o \
    $(OBJDIR)/ScaledLabelScorer.o \
    $(OBJDIR)/ScoreAccessor.o \
    $(OBJDIR)/ScoringContext.o \
    $(OBJDIR)/ScaledLabelScorer.o \
    $(OBJDIR)/StatefulOnnxLabelScorer.o \
    $(OBJDIR)/TransitionLabelScorer.o \
    $(OBJDIR)/TransitionTypes.o

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLabelScorer.$(a): $(LIBSPRINTLABELSCORER_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLABELSCORER_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
