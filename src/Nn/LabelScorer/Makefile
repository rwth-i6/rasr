#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLabelScorer.$(a)

LIBSPRINTLABELSCORER_O =  \
    $(OBJDIR)/BufferedLabelScorer.o \
    $(OBJDIR)/CombineLabelScorer.o \
    $(OBJDIR)/DataView.o \
    $(OBJDIR)/Encoder.o \
    $(OBJDIR)/EncoderDecoderLabelScorer.o \
    $(OBJDIR)/EncoderFactory.o \
	$(OBJDIR)/FullInputStatefulOnnxLabelScorer.o \
    $(OBJDIR)/LabelScorer.o \
    $(OBJDIR)/LabelScorerFactory.o \
    $(OBJDIR)/FixedContextOnnxLabelScorer.o \
    $(OBJDIR)/NoContextOnnxLabelScorer.o \
    $(OBJDIR)/NoOpLabelScorer.o \
    $(OBJDIR)/OnnxEncoder.o \
    $(OBJDIR)/ScoringContext.o \
    $(OBJDIR)/StatefulOnnxLabelScorer.o

CCFLAGS += -fexceptions


# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLabelScorer.$(a): $(LIBSPRINTLABELSCORER_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLABELSCORER_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
