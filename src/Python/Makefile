#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

CCFLAGS		+= -fexceptions

ifdef MODULE_TENSORFLOW
CXXFLAGS += $(TF_CXXFLAGS)
LDFLAGS  += $(TF_LDFLAGS)
endif

# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= libSprintPython.$(a) check$(exe)

LIBPYTHON_O    = \
		$(OBJDIR)/AllophoneStateFsaBuilder.o \
		$(OBJDIR)/Configuration.o            \
		$(OBJDIR)/Init.o                     \
		$(OBJDIR)/Numpy.o                    \
		$(OBJDIR)/LabelScorer.o              \
		$(OBJDIR)/ScoringContext.o           \
		$(OBJDIR)/Search.o                   \
		$(OBJDIR)/Utilities.o

CHECK_O		= $(OBJDIR)/check.o 			\
		  libSprintPython.$(a)			\
		  ../Math/libSprintMath.$(a)		\
		  ../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintPython.$(a): $(LIBPYTHON_O)
	$(MAKELIB) $@ $^

check$(exe): $(CHECK_O)
	$(LD) $^ -o check$(exe) $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBPYTHON_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
