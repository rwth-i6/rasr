#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintLabelScorer.$(a)

LIBSPRINTLABELSCORER_O =  \
		$(OBJDIR)/BufferedLabelScorer.o \
		$(OBJDIR)/Encoder.o \
		$(OBJDIR)/EncoderDecoderLabelScorer.o \
		$(OBJDIR)/LabelScorer.o \
		$(OBJDIR)/ScoringContext.o \

CCFLAGS += -fexceptions

ifdef MODULE_ONNX
    LIBSPRINTLABELSCORER_O += $(OBJDIR)/LimitedCtxOnnxLabelScorer.o \
							  $(OBJDIR)/NoCtxOnnxLabelScorer.o \
							  $(OBJDIR)/OnnxEncoder.o \
							  $(OBJDIR)/StatefulOnnxLabelScorer.o

	  CHECK_O += ../Onnx/libSprintOnnx.$(a)
endif

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintLabelScorer.$(a): $(LIBSPRINTLABELSCORER_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTLABELSCORER_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
