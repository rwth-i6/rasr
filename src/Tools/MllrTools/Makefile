#!make

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= 

MEAN_SUMFILE_CREATOR_O = $(OBJDIR)/MeanSumfileEstimator.o \
		  ../../Bliss/libSprintBliss.$(a) \
		  ../../Flow/libSprintFlow.$(a) \
		  ../../Mm/libSprintMm.$(a) \
		  ../../Core/libSprintCore.$(a)

ifdef MODULE_CART
MEAN_SUMFILE_CREATOR_O += ../../Cart/libSprintCart.$(a)
endif
ifdef MODULE_PYTHON
MEAN_SUMFILE_CREATOR_O += ../../Python/libSprintPython.$(a)
endif

# -----------------------------------------------------------------------------

build: $(TARGETS)
install:

#MeanSumfileEstimator$(exe): $(MEAN_SUMFILE_CREATOR_O)
#	$(LD) $(MEAN_SUMFILE_CREATOR_O) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(MEAN_SUMFILE_CREATOR_O)))
