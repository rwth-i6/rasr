"""
Analog Plug-in for creating nist-stm and nist-ctm formated files.
"""

__version__ = '$Revision$'
__date__ = '$Date$'


import string, sys
from analog_util.analog import Writer


def nistRecordingAndTrack(segment, useName=False):
    if useName:
        rec = segment['name']
    else:
        rec = segment['recording']

    track = segment['track'] + 1
    return '%s %d' % (rec, track)

class NistStmList(Writer):
    id = 'nist-stm'
    defaultPostfix = '.stm'

    def __call__(self, file, data):
        for segment in data:
            print(nistRecordingAndTrack(segment), \
                segment.get('speaker', 'unknown'), \
                segment['start'], \
                segment['end'], \
                string.join(string.split(segment['reference']), ' '), file=file)

class NistStmRecList(Writer):
    id = 'nist-stm-rec'
    defaultPostfix = '.stm'

    def __call__(self, file, data):
        for segment in data:
            print(nistRecordingAndTrack(segment), \
                segment.get('speaker', 'unknown'), \
                segment['start'], \
                segment['end'], \
                string.join(string.split(segment['recognized']), ' '), file=file)

class NistCtmList(Writer):
    id = 'nist-ctm'
    defaultPostfix = '.ctm'

    def __init__(self, options):
        super(NistCtmList, self).__init__(options)
        self.frameShift = float(options.frameShift)
        self.silence = options.silenceLemma
        self.useFullName = options.fullName

    def __call__(self, file, data):
        for segment in data:
            recording = nistRecordingAndTrack(segment, self.useFullName)
            segmentStartTime = float(segment['start'])
            traceback = segment['traceback']
            previousFrameIndex = 0
            for frameIndex, score, lemma in traceback[1:-1]:
                frameIndex = int(frameIndex)
                if lemma == self.silence : lemma = "@"
                print(recording, \
                      "%.3f" % (segmentStartTime + float(previousFrameIndex) * self.frameShift), \
                      "%.3f" % (float(frameIndex - previousFrameIndex) * self.frameShift), \
                      lemma, file=file)
                previousFrameIndex = frameIndex
