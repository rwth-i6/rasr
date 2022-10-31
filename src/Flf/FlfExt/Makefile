#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= libSprintFlfExt.$(a)

# check$(exe)

LIBSPRINTFLFEXT_O = \
		  $(OBJDIR)/AcousticAlignment.o \
		  $(OBJDIR)/MapDecoder.o \
		  $(OBJDIR)/MtConfusionNetwork.o \
		  $(OBJDIR)/WindowedLevenshteinDistanceDecoder.o

# CHECK_O		= $(OBJDIR)/check.o \
# 		  libSprintFlfExt.$(a) \
# 		  ../../Flf/libSprintFlf.$(a) \
# 		  ../../Fsa/libSprintFsa.$(a) \
# 		  ../../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintFlfExt.$(a): $(LIBSPRINTFLFEXT_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $^ -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTFLFEXT_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
