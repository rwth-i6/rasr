# Generate code for AMD 64 bit processors

ifdef _ADD_OPENFST
ifeq ($(shell test -d $(OPENFSTDIR)/lib64 && echo 1),1)
LDFLAGS 	+= -L$(OPENFSTDIR)/lib64 $(OPENFSTLIBS)
else
LDFLAGS 	+= -L$(OPENFSTDIR)/lib -Wl,-rpath=$(OPENFSTDIR)/lib $(OPENFSTLIBS)
endif
endif


ifeq ($(COMPILER),gcc)
CCFLAGS		+= -ffast-math
# CCFLAGS     += -mfpmath=sse
# CCFLAGS     += -funroll-loops
CCFLAGS     += -msse3
CCFLAGS     += -march=native
endif


ifneq ($(COMPILER),sun)
ifeq ($(COMPILE),standard)
CCFLAGS		+= -O2
endif
ifeq ($(COMPILE),release)
CCFLAGS		+= -O3
endif
ifeq ($(COMPILE),debug)
CCFLAGS		+= -O0
endif
else # sun studio compiler
CCFLAGS         += -m64
endif # COMPILER

ifeq ($(PROFILE),valgrind)
DEFINES		+= -DDISABLE_SIMD
endif
