# Project:      SPRINT
# File:		Config.make - auto-detected part of build environment
# Revision:     $Id$

# Look into Options.make for configurable settings !!!

COMPILE_MODES		= standard release debug debug_light debug_dynamic
PROFILE_MODES		= none gprof valgrind
COMPILERS               = gcc icc clang

include $(TOPDIR)/Options.make
include $(TOPDIR)/Modules.make

# -----------------------------------------------------------------------------
# sanity check on configurable settings

ifeq (,$(findstring $(COMPILE), $(COMPILE_MODES)))
$(error meaningful values for COMPILE are $(COMPILE_MODES))
endif

ifneq ($(shell test $(DBG_LEVEL) -ge -1; echo $$?),0)
$(error DBG_LEVEL=$(DBG_LEVEL), but meaningful values for DBG_LEVEL are [-1,0,1,...,N])
endif

ifeq (,$(findstring $(PROFILE), $(PROFILE_MODES)))
$(error meaningful values for PROFILE are $(PROFILE_MODES))
endif
ifeq (,$(findstring $(COMPILER), $(COMPILERS)))
$(error supported values for COMPILER are $(COMPILERS))
endif

# -----------------------------------------------------------------------------
# auto-detected processor and operating system settings

PROC = $(shell uname -m)
OS   = $(shell uname -s)
CPU  = $(shell test -e /proc/cpuinfo && cat /proc/cpuinfo | grep "model name" | head -n 1)
ifeq ($(CPU),)
CPU = unknown
endif

OS_LIST		= linux darwin
ifeq ($(OS),Linux)
OS		= linux
endif
ifeq ($(OS),Darwin)
OS		= darwin
endif

# -----------------------------------------------------------------------------
# preprocesor defines

ARCH		= $(OS)_$(PROC)
ifeq ($(COMPILE),standard)
DEFINES         += -DSPRINT_STANDARD_BUILD
endif
ifeq ($(COMPILE),release)
DEFINES		+= -DSPRINT_RELEASE_BUILD
DEFINES		+= -DNDEBUG
endif
ifeq ($(COMPILE),debug)
DEFINES         += -DDEBUG
endif
ifeq ($(COMPILE),debug_light)
DEFINES         += -DSPRINT_STANDARD_BUILD
DEFINES         += -DSPRINT_DEBUG_LIGHT
endif
ifeq ($(COMPILE),debug_dynamic)
DEFINES         += -DSPRINT_STANDARD_BUILD
DEFINES         += -DSPRINT_DEBUG_LIGHT
endif

DEFINES     += -DDBG_LEVEL=$(DBG_LEVEL)

DEFINES		+= -DPROC_$(PROC) -DOS_$(OS) -DARCH_$(ARCH)

CPPFLAGS	+= $(DEFINES)

# -----------------------------------------------------------------------------
# Sprint includes

INCLUDES	= -I. -I$(TOPDIR)/src

CPPFLAGS	+= $(INCLUDES)

# -----------------------------------------------------------------------------
# Where and how to store object files

ifeq ($(PROFILE),none)
ifneq ($(SUFFIX),)
OBJEXT	= $(OS)-$(PROC)-$(COMPILE)-$(SUFFIX)
else
OBJEXT	= $(OS)-$(PROC)-$(COMPILE)
endif # SUFFIX
else
OBJEXT	= $(OS)-$(PROC)-$(COMPILE)-$(PROFILE)
DEFINES	+= -DPROFILE_$(PROFILE)
endif # PROFILE

DLIBEXT = so

ifeq ($(COMPILE),debug_dynamic)
LIBEXT	= $(DLIBEXT)
else
LIBEXT	= a
endif

OBJDIR	= .build/$(OBJEXT)
ifeq ($(BINARYFILENAMES),extended)
exe	= .$(OBJEXT)
a	= $(OBJEXT).$(LIBEXT)
else
exe	=
a	= $(LIBEXT)
endif

so      = .$(DLIBEXT)

# -----------------------------------------------------------------------------
# laod compiler specifc settings
include $(TOPDIR)/config/cc-$(COMPILER).make

# load OS specific settings
include $(TOPDIR)/config/os-$(OS).make

# load processor specific settings
include $(TOPDIR)/config/proc-$(PROC).make

ifneq ($(ADDCCFLAGS),)
CCFLAGS += $(ADDCCFLAGS)
endif

# -----------------------------------------------------------------------------
# enable compiler cache

ifneq ($(shell which ccache),)
ifneq ($(shell hostname | cut -f 1 -d -),cluster)
CCACHE		= ccache
CCACHE_UNIFY	= 1
CC		:= $(CCACHE) $(CC)
CXX		:= $(CCACHE) $(CXX)

# share compiler cache inside i6
ifeq ($(shell test -d /u/ccache; echo $$?),0)
CCACHE_DIR	= /u/ccache
CCACHE_UMASK	= 000
endif # i6
endif # cluster
endif # ccache

# -----------------------------------------------------------------------------
# common tools

LEX		= flex
YACC		= bison -d
LATEX		= latex
DVIPS		= dvips
MAKEINDEX	= makeindex
