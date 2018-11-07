#!gmake

TOPDIR		= ../../..

include $(TOPDIR)/Makefile.cfg

# -----------------------------------------------------------------------------

SUBDIRS 	=
TARGETS		= ger.py

# -----------------------------------------------------------------------------

build: $(TARGETS)

install:
	rm -f $(INSTALL_TARGET)/ger.py
	ln -s `pwd`/ger.py $(INSTALL_TARGET)/ger.py

ger.py:
	echo "Error: Could not find ger.py" >&2  

include $(TOPDIR)/Rules.make
