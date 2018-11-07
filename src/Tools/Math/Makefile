#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS		=
TARGETS		= matrix-tool$(exe) 

MATRIX_TOOL_O	= $(OBJDIR)/MatrixTool.o \
		../../Math/libSprintMath.$(a) \
		../../Math/Nr/libSprintMathNr.$(a) \
		../../Math/Lapack/libSprintMathLapack.$(a) \
	    	../../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

build: $(TARGETS)

matrix-tool$(exe): $(MATRIX_TOOL_O)
	$(LD) $^ -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(patsubst %.o,%.d,$(filter %.o,$(MATRIX_TOOL_O)))
