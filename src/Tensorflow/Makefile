#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)

SUBDIRS		=
TARGETS		= libSprintTensorflow.$(a) check$(exe)

LIBSPRINTTENSORFLOW_O = $(OBJDIR)/DummyStateManager.o \
                        $(OBJDIR)/Graph.o \
                        $(OBJDIR)/GraphLoader.o \
                        $(OBJDIR)/MetaGraphLoader.o \
                        $(OBJDIR)/Module.o \
                        $(OBJDIR)/Session.o \
                        $(OBJDIR)/StateManager.o \
                        $(OBJDIR)/Tensor.o \
                        $(OBJDIR)/TensorflowForwardNode.o \
                        $(OBJDIR)/TensorflowFeatureScorer.o \
                        $(OBJDIR)/TensorflowFeatureScorerStateCarryover.o \
                        $(OBJDIR)/TensorMap.o \
                        $(OBJDIR)/TransformerStateManager.o \
                        $(OBJDIR)/VanillaGraphLoader.o

CHECK_O = $(OBJDIR)/check.o \
          libSprintTensorflow.$(a) \
          ../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintTensorflow.$(a): $(LIBSPRINTTENSORFLOW_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTTENSORFLOW_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))

