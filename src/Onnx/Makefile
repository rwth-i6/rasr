#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintOnnx.$(a) check$(exe)

LIBSPRINTONNX_O = $(OBJDIR)/Session.o \
				  $(OBJDIR)/IOSpecification.o \
				  $(OBJDIR)/Model.o \
				  $(OBJDIR)/Module.o \
				  $(OBJDIR)/OnnxEncoder.o \
				  $(OBJDIR)/OnnxFeatureScorer.o \
				  $(OBJDIR)/OnnxForwardNode.o \
                  $(OBJDIR)/Value.o

CHECK_O = $(OBJDIR)/check.o \
          libSprintOnnx.$(a) \
          ../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintOnnx.$(a): $(LIBSPRINTONNX_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTONNX_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))

