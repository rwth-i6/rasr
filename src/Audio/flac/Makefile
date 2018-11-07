#!gmake

TOPDIR          = ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS         =
TARGETS         = 

FLAC_O       = $(OBJDIR)/FlacDecoder.o

# -----------------------------------------------------------------------------

all: $(FLAC_O)

include $(TOPDIR)/Rules.make

# -----------------------------------------------------------------------------
# dependencies
# DO NOT DELETE

FlacDecoder.o: ./FlacDecoder.hh
