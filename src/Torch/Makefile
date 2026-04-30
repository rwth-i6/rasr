#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintTorch.$(a) check$(exe)

LIBSPRINTTORCH_O = \
        $(OBJDIR)/ConformerStateManager.o \
        $(OBJDIR)/DummyStateManager.o \
        $(OBJDIR)/IOSpecification.o \
        $(OBJDIR)/IOSpecParser.o \
        $(OBJDIR)/Model.o \
        $(OBJDIR)/Module.o \
        $(OBJDIR)/Session.o \
        $(OBJDIR)/StateManager.o \
        $(OBJDIR)/TorchEncoder.o \
        $(OBJDIR)/TorchFeatureScorer.o \
        $(OBJDIR)/TorchForwardNode.o \

CHECK_O = $(OBJDIR)/check.o \
          ../Core/libSprintCore.$(a) \
          ../Nn/libSprintNn.$(a) \
          ../Onnx/libSprintOnnx.$(a) \
          libSprintTorch.$(a) \
          ../DropDead/libDropDead.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintTorch.$(a): $(LIBSPRINTTORCH_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTTORCH_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))