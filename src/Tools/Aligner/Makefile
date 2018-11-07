#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

OBJDIR = $(BUILDDIR)/$(OBJEXT)/Aligner

# -----------------------------------------------------------------------------

CXXFLAGS += -I$(TOPDIR)/src/Translation/SRI
LDFLAGS += -loolm \
			-ldstruct \
			-lmisc

SUBDIRS 	=
TARGETS		= aligner$(exe)

ALIGNER_O	= $(OBJDIR)/Aligner.o \
				$(OBJDIR)/AlignAutomaton.o \
				$(OBJDIR)/SimpleAlignAutomaton.o \
				$(OBJDIR)/ZeroOrderAlignAutomaton.o \
				$(OBJDIR)/Common.o \
				$(OBJDIR)/ConditionalLexicon.o \
				$(OBJDIR)/ConditionalLexiconPlain.o \
				$(OBJDIR)/ConditionalLexiconSri.o \
				../Translation/libSprintTranslation.$(a) \
				../Fsa/libSprintFsa.$(a) \
				../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	$(INSTALL) $(TARGETS) $(INSTALL_TARGET)

aligner$(exe): $(ALIGNER_O)
	$(LD) $^ -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(ALIGNER_O)))
