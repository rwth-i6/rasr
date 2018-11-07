#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

CCFLAGS += -Wno-sign-compare

ifeq ($(OS),darwin)
CCFLAGS         += -fexceptions
endif


# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= libSprintOpenFst.$(a) check$(exe)

LIB_OPEN_FST_O    = $(OBJDIR)/Input.o \
					$(OBJDIR)/Module.o \
					$(OBJDIR)/Output.o \
					$(OBJDIR)/SymbolTable.o

CHECK_O		= $(OBJDIR)/check.o			\
		  libSprintOpenFst.$(a)			\
		  ../Bliss/libSprintBliss.$(a)  \
		  ../Fsa/libSprintFsa.$(a)		\
		  ../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

libSprintOpenFst.$(a): $(LIB_OPEN_FST_O)
	$(MAKELIB) $@ $^

check$(exe): libSprintOpenFst.$(a) $(CHECK_O)   
	$(LD) -Wl,-\( $^ -Wl,-\) -o $@ $(LDFLAGS)
	
include $(TOPDIR)/Rules.make

sinclude $(LIB_OPEN_FST_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
