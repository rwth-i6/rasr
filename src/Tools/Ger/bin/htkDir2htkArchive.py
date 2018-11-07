#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

if __name__ == '__main__':
    import os, sys
    binDir = sys.argv[0]
    while os.path.islink(binDir):
	binDir = os.path.join(os.path.dirname(binDir), os.readlink(binDir))
    binDir = os.path.dirname(binDir)
    sys.path = [binDir, os.path.join(os.path.join(binDir, '..'), 'lib')] + sys.path

import os
import os.path
import shutil
import sys
from miscLib import mktree
from blissLib import BlissCorpusParser
from ioLib import uopen, uclose

def getSuffix(path):
    lastIndex = len(path)
    if path.endswith('.tar.gz'):
	lastIndex -= 7
    elif path.endswith('.gz'):
	lastIndex -= 3
    i = path.rfind('.', 0, lastIndex)
    if i == -1:
	return path[lastIndex:]
    else:
	return path[i:]

class HtkSlfLatticeArchiver(BlissCorpusParser):
    def __init__(self, latticeArchiveDir, latticeDir, latticeEncoding, latticeSuffix = '.lat.gz', isCopy = False):
	BlissCorpusParser.__init__(self)
	self.parseSpeakerAndCondition(False)
	self.latticeArchiveDir = latticeArchiveDir
	self.latticeDir = latticeDir
	self.latticeEncoding = latticeEncoding
	self.latticeSuffix = latticeSuffix
	if isCopy:
	    self.transfer = self.copy
	else:
	    self.transfer = self.link
	self.corpusName = None
	self.corpusDir = None
	self.recordingName = None
	self.recordingDir = None
	self.segmentCounter = 0
	self.latticeName = None

	self.isLog = False

    def setLog(self, isLog):
	self.isLog = isLog

    def link(self, latticePath, archivedLatticePath):
	os.symlink(os.path.abspath(latticePath), archivedLatticePath)
	if self.isLog:
	    print latticePath, '--ln-->', archivedLatticePath

    def copy(self, latticePath, archivedLatticePath):
	shutil.copy(latticePath, archivedLatticePath)
	if self.isLog:
	    print latticePath, '--cp-->', archivedLatticePath

    def startCorpus(self, attr):
	self.corpusName = attr.get('name', '')
	self.corpusDir = os.path.join(self.latticeArchiveDir, self.corpusName)

    def endMain(self):
	defaultConfigPath = os.path.join(self.latticeArchiveDir, 'default.config')
	fd = uopen(defaultConfigPath, 'utf-8', 'w')
	self.writeDefaultConfig(fd)
	uclose(fd)
	if self.isLog:
	    print >> sys.stderr, '-->', defaultConfigPath

    def startRecording(self, attr):
	self.recordingName = attr.get('name', '')
	self.recordingDir = os.path.join(self.corpusDir, self.recordingName)
	mktree(self.recordingDir)
	self.segmentCounter = 1
	self.changeRecording(attr.get('name', ''), attr['audio'])

    def startSegment(self, attr):
	segmentName = attr.get('name', str(self.segmentCounter))
	self.changeSegment(segmentName, float(attr.get('start', '0.0')), float(attr.get('end', 'inf')))
	latticePath = self.getLattice()
	if os.path.exists(latticePath):
	    archivedLatticePath = os.path.join(self.recordingDir, segmentName + getSuffix(latticePath))
	    self.transfer(latticePath, archivedLatticePath)
	    self.checkLattice(latticePath)
	else:
	    print >> sys.stderr, 'Error:', latticePath, 'does not exist'
	self.segmentCounter += 1

    def changeRecording(self, recName, audioFile):
	pass

    def changeSegment(self, segmentName, startTime, endTime):
	self.latticeName = segmentName + self.latticeSuffix

    def getLattice(self):
	return os.path.join(self.latticeDir, self.latticeName)

    def checkLattice(self, latticePath):
	pass

    def writeDefaultConfig(self, fd):
	print >> fd, '[*.lattice.archive.reader]'
	print >> fd, '%-16s = %s' % ('encoding', self.latticeEncoding)
	print >> fd, '%-16s = %s' % ('suffix',   self.latticeSuffix)


if __name__ == '__main__':
    from optparse import OptionParser
    optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <bliss-corpus>'
    )
    optparser.add_option("-c", "--copy", dest="copy", action="store_true", default=False,
			 help="do not create symbolic, but copy lattices")
    optparser.add_option("-a", "--archive-dir", dest="latticeArchiveDir", default="archive",
			 help="htk slf lattice archive directory", metavar="DIR")
    optparser.add_option("-l", "--lattice-dir", dest="latticeDir", default="",
			 help="htk slf lattice directory", metavar="DIR")
    optparser.add_option("-s", "--suffix", dest="suffix", default=".lat.gz",
			 help="htk slf lattice suffix; default is '.lat.gz'", metavar="SUFFIX")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
			 help="produce verbose output")
    optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
			 help="encoding of htk lattices; default is 'ascii'", metavar="ENCODING")

    if len(sys.argv) == 1:
	optparser.print_help()
	sys.exit(0)
    options, args = optparser.parse_args()

    # archive
    archiver = HtkSlfLatticeArchiver(options.latticeArchiveDir, options.latticeDir, options.encoding, options.suffix, options.copy)
    archiver.setLog(options.verbose)
    archiver.parse(args[0])
