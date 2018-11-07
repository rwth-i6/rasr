#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= fsa$(exe)

FSA_O		= $(OBJDIR)/Fsa.o \
		  ../../Fsa/libSprintFsa.$(a) \
		  ../../Core/libSprintCore.$(a)

ifdef MODULE_OPENFST
FSA_O += ../../OpenFst/libSprintOpenFst.$(a)
endif

#		  ../../Lattice/libSprintLattice.$(a)

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

fsa$(exe): $(FSA_O)
	$(LD) $^ -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(FSA_O)))
