#!gmake
# -----------------------------------------------------------------------------
# project:    SPRINT
# -----------------------------------------------------------------------------

default:	build

TOPDIR = ..

include $(TOPDIR)/Config.make

# -----------------------------------------------------------------------------
SUBDIRS		= Am \
		  Audio \
		  Bliss \
		  Core \
		  Flow \
		  Fsa \
		  Lattice \
		  Lm \
		  Math \
		  Mc \
		  Mm \
		  Search \
		  Signal \
		  Speech \
		  Tools

# dependencies of module Speech
SPEECH_DEP = Am Flow Lm Mm Search Mc
# dependencies of module Am
AM_DEP     = Core Bliss Mm

BASIC_MODULES = Am Audio Bliss Core Flow Fsa Lattice Lm Math Mm Signal Speech
TOOLS_MODULES = $(BASIC_MODULES)


ifdef MODULE_CART
SUBDIRS         += Cart
TOOLS_MODULES   += Cart
SPEECH_DEP      += Cart
AM_DEP          += Cart
endif

ifdef MODULE_FLF_CORE
SUBDIRS         += Flf/FlfCore
BASIC_MODULES   += FlfCore
TOOLS_MODULES   += FlfCore
endif

ifdef MODULE_FLF_EXT
SUBDIRS         += Flf/FlfExt
TOOLS_MODULES   += FlfExt
endif

ifdef MODULE_FLF
SUBDIRS         += Flf
TOOLS_MODULES   += Flf
endif

ifdef MODULE_OPENFST
SUBDIRS       += OpenFst
TOOLS_MODULES += OpenFst
endif

ifdef MODULE_NN
SUBDIRS       += Nn
BASIC_MODULES += Nn
TOOLS_MODULES += Nn
endif

ifdef MODULE_ONNX
SUBDIRS       += Onnx
TOOLS_MODULES += Onnx
endif

ifdef MODULE_PYTHON
PYTHON_AS_DEP = Python
SUBDIRS       += Python
BASIC_MODULES += Python
TOOLS_MODULES += Python
else
PYTHON_AS_DEP =
endif

ifdef MODULE_TENSORFLOW
SUBDIRS       += Tensorflow
TOOLS_MODULES += Tensorflow
endif

ifdef MODULE_TEST
SUBDIRS         += Test
endif

TARGETS		=

# -----------------------------------------------------------------------------

build:	source-version $(SUBDIRS)

include $(TOPDIR)/Rules.make

source-version:
	@# If a file named SourceVersion.release is available then use this for
	@# version information (for released versions of sprint).
	@# If machine chosen for compilation does not support svn or svn version
	@# is not up-to-date then discard svn version info.
	@if [ -f SourceVersion.release ]; then \
		cp SourceVersion.release .SourceVersion.tmp ;\
	elif svn info > /dev/null 2>&1; then \
		svn info          > SourceVersion ;\
		svn status | sort >> SourceVersion ;\
		awk '{printf "\"%s\\n\"\n", $$0}' SourceVersion > .SourceVersion.tmp ;\
	elif ../scripts/git-version.py > /dev/null 2>&1; then \
		../scripts/git-version.py > SourceVersion ;\
		awk '{printf "\"git-%s\\n\"\n", $$0}' SourceVersion > .SourceVersion.tmp ;\
	else \
		/bin/echo '"n/a\n"' > .SourceVersion.tmp ;\
	fi ; \
	diff -I "SourceVersion.tmp" SourceVersion.cc .SourceVersion.tmp >/dev/null 2>&1 ; \
	if [ $$? -ne 0  ]; then \
		mv .SourceVersion.tmp SourceVersion.cc ;\
	else \
		rm -f .SourceVersion.tmp; \
	fi;

.PHONY:	source-version

modules: $(TOOLS_MODULES)

.PHONY: modules

install:
	$(MAKE) -C Tools $@

.PHONY: $(SUBDIRS)


doc:
	$(MAKE) -C $@

Am: $(AM_DEP)
	$(MAKE) -C Am libSprintAm.$(a)

Audio: Flow
	$(MAKE) -C Audio libSprintAudio.$(a)

Bliss: Core $(PYTHON_AS_DEP)
	$(MAKE) -C Bliss libSprintBliss.$(a)

Cart: Core
	$(MAKE) -C Cart libSprintCart.$(a)

Core: source-version Modules.hh
	$(MAKE) -C Core libSprintCore.$(a)

Flow: Core
	$(MAKE) -C Flow libSprintFlow.$(a)

Flf: Speech
	$(MAKE) -C Flf libSprintFlf.$(a)

FlfCore : Core
	$(MAKE) -C Flf/FlfCore libSprintFlfCore.$(a)

FlfExt : Core
	$(MAKE) -C Flf/FlfExt libSprintFlfExt.$(a)

Fsa: Core
	$(MAKE) -C Fsa libSprintFsa.$(a)

Lattice: Core Fsa
	$(MAKE) -C Lattice libSprintLattice.$(a)

Lm: Core Bliss
	$(MAKE) -C Lm libSprintLm.$(a)

Math: Core
	$(MAKE) -C Math

Mm: Core
	$(MAKE) -C Mm libSprintMm.$(a)

Mc: Core
	$(MAKE) -C Mc libSprintMc.$(a)

Me: Core Mm Signal
	$(MAKE) -C Me libSprintMe.$(a)

Nn: Core Flow Math Signal $(PYTHON_AS_DEP)
	$(MAKE) -C Nn libSprintNn.$(a)

OpenFst: Core Fsa
	$(MAKE) -C OpenFst libSprintOpenFst.$(a)

Onnx: Core Math
	$(MAKE) -C Onnx libSprintOnnx.$(a)

Python: Core Math
	$(MAKE) -C Python libSprintPython.$(a)

Search: Core Bliss Fsa Lattice
	$(MAKE) -C Search 

Signal: Flow Math
	$(MAKE) -C Signal libSprintSignal.$(a)

Sparse: Speech
	$(MAKE) -C Sparse libSprintSparse.$(a)

Speech: $(SPEECH_DEP)
	$(MAKE) -C Speech libSprintSpeech.$(a)

Tensorflow : Core Math
	$(MAKE) -C Tensorflow

Test: $(TOOLS_MODULES)
	$(MAKE) -C Test

Tools: $(TOOLS_MODULES)
	$(MAKE) -C Tools

# -----------------------------------------------------------------------------
tags:
	find . -regex '.*\.[hc][hc]?' | xargs etags

.PHONY:	tags
