#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= libSprintTreeBuilder.$(a)

LIBSPRINTTREEBUILDER_O = $(OBJDIR)/PersistentStateTree.o \
                         $(OBJDIR)/StateTree.o \
                         $(OBJDIR)/StateTreeIo.o \
                         $(OBJDIR)/TreeBuilder.o \
                         $(OBJDIR)/TreeStructure.o

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintTreeBuilder.$(a): $(LIBSPRINTTREEBUILDER_O)
	$(MAKELIB) $@ $^

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTTREEBUILDER_O:.o=.d)
include $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))