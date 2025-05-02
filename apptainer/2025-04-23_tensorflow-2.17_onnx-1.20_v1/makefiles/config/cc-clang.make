# LLVM/Clang Compiler Settings

BINDIR          =
GCC_VERSION	=

# -----------------------------------------------------------------------------
# Compiler
CC		= $(BINDIR)clang$(GCC_VERSION)
LD              = $(BINDIR)clang++$(GCC_VERSION)
CXX             = $(BINDIR)clang++$(GCC_VERSION)
CXXLD           = $(BINDIR)clang++$(GCC_VERSION)

CPPFLAGS        += $(SYS_INCLUDES)

CCFLAGS		= 		# common for C and C++
CXXFLAGS        = $(CCFLAGS)    # options for C++ compiler
CFLAGS		= $(CCFLAGS)	# options for C compiler

CXX_MAJOR = $(shell $(CXX) --version | head -n 1 | sed -e 's/.*[ \t]\([0-9]\)\.\([0-9]\)\.\([0-9]\)\([ \t].*\)*$$/\1/')
CXX_MINOR = $(shell $(CXX) --version | head -n 1 | sed -e 's/.*[ \t]\([0-9]\)\.\([0-9]\)\.\([0-9]\)\([ \t].*\)*$$/\2/')

# -----------------------------------------------------------------------------
# compiler options
DEFINES		+= -D_GNU_SOURCE
DEFINES		+= -D_GLIBCXX_PERMIT_BACKWARD_HASH
CCFLAGS		+= -pipe
CCFLAGS		+= -funsigned-char
CFLAGS		+= -std=c99
CXXFLAGS	+= -std=gnu++0x
ifeq ($(OS),darwin)
CXXFLAGS	+= -stdlib=libc++
else
CXXFLAGS	+= -D__float128=void  # hack: http://llvm.org/bugs/show_bug.cgi?id=13530
endif
#CCFLAGS	+= -pedantic
CCFLAGS		+= -Wall
CCFLAGS		+= -Wno-long-long
#CXXFLAGS	+= -Woverloaded-virtual
#CFLAGS     += -Weffc++
#CFLAGS		+= -Wold-style-cast
#CCFLAGS         += -pg
#LDFLAGS         += -pg
ifdef MODULE_OPENMP
CCFLAGS		+= -fopenmp
LDFLAGS		+= -fopenmp
CPPFLAGS	+= -I/usr/lib/gcc/x86_64-linux-gnu/4.6/include/
endif

ifeq ($(strip $(CXX_MAJOR)),4)
ifeq ($(shell test $(CXX_MINOR) -ge 3 && echo 1),1)
# gcc >= 4.3

# code uses ext/hash_map, ext/hash_set etc.
CXXFLAGS += -Wno-deprecated

# strict type based alias analysis doesn't work with our implementation of
# reference counting smart pointers (Core::Ref)
CXXFLAGS += -fno-strict-aliasing
endif
endif

ifeq ($(COMPILE),debug)
CCFLAGS		+= -g
DEFINES		+= -D_GLIBCXX_DEBUG
endif

ifeq ($(COMPILE),debug_light)
CCFLAGS		+= -g
endif

ifeq ($(COMPILE),debug_dynamic)
CFLAGS		+= -g -fPIC
CCFLAGS		+= -g -fPIC
LDFLAGS		+= -Wl,--allow-shlib-undefined
endif

ifneq ($(COMPILE),release)
# needed to get symbolic function names in stack traces (see Core/Assertions.cc)
LDFLAGS          += -rdynamic
endif

ifeq ($(PROFILE),bprof)
CCFLAGS		+= -g
LDFLAGS		+= /usr/lib/bmon.o
PROF		= bprof
endif
ifeq ($(PROFILE),gprof)
CCFLAGS		+= -pg
LDFLAGS		+= -pg  -static
PROF		= gprof -b
endif
ifeq ($(PROFILE),valgrind)
CCFLAGS		+= -g
LDFLAGS		+=
PROF		= echo "type: valgrind <execuable>"
endif
ifeq ($(PROFILE),purify)
PROFILE		+= "(not supported)"
PROF		=
endif
