#!gmake

TOPDIR		= ../..

include $(TOPDIR)/Makefile.cfg

LIBSPRINTAUDIO_O	= $(OBJDIR)/Module.o $(OBJDIR)/Node.o

ifdef MODULE_AUDIO_FFMPEG
LIBSPRINTAUDIO_O += $(OBJDIR)/Ffmpeg.o
endif
ifdef MODULE_AUDIO_FLAC
SUBDIRS		 += flac
LIBSPRINTAUDIO_O += $(OBJDIR)/Flac.o
endif
ifeq ($(OS),linux)
ifdef MODULE_AUDIO_OSS
LIBSPRINTAUDIO_O	+= $(OBJDIR)/Oss.o
endif
endif
ifdef MODULE_AUDIO_RAW
LIBSPRINTAUDIO_O += $(OBJDIR)/Raw.o
endif
ifdef MODULE_AUDIO_WAV_SYSTEM
LIBSPRINTAUDIO_O += $(OBJDIR)/Wav.o
endif


# -----------------------------------------------------------------------------


TARGETS		= libSprintAudio.$(a) check$(exe)

CHECK_O		= $(OBJDIR)/check.o		\
		  libSprintAudio.$(a)		\
		  ../Flow/libSprintFlow.$(a)	\
		  ../Core/libSprintCore.$(a)

# -----------------------------------------------------------------------------

all: $(TARGETS)

flac:
	$(MAKE) -C $@

.PHONY:	$(SUBDIRS)

libSprintAudio.$(a): $(SUBDIRS) $(LIBSPRINTAUDIO_O)
	$(MAKELIB) $@ $(LIBSPRINTAUDIO_O) $(patsubst %,%/$(OBJDIR)/*.o,$(SUBDIRS))

check$(exe): $(CHECK_O)
	$(LD) $(CHECK_O) -o $@ $(LDFLAGS)

include $(TOPDIR)/Rules.make

sinclude $(LIBSPRINTAUDIO_O:.o=.d)
sinclude $(patsubst %.o,%.d,$(filter %.o,$(CHECK_O)))
