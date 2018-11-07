#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

if __name__ == '__main__':
    import os, sys
    binDir = sys.argv[0]
    while os.path.islink(binDir):
	binDir = os.path.join(os.path.dirname(binDir), os.readlink(binDir))
    binDir = os.path.dirname(binDir)
    sys.path = [binDir, os.path.join(os.path.join(binDir, '..'), 'lib')] + sys.path

import os.path
import re
import sys
from blissLib import BlissCorpusParser
from ioLib import uopen, uclose
from xmlWriterLib import openXml, closeXml

class HtkLatticeOrthographyExtractor:
    def __init__(self, addOrth):
	self.addOrth = addOrth

    def extract(self, fd):
	for line in fd:
	    i = line.find('#')
	    if i > -1:
		line = line[:i]
	    for token in line.split():
		if token.startswith('W='):
		    self.addOrth(token[2:])


class BlissOrthographyExtractor:
    def __init__(self, addOrth):
	self.addOrth = addOrth

    reBlissToken = re.compile(r'(?:<.+?>)|(?:\[.+?\])|(?:\S+)')
    def extract(self, s):
	for token in self.reBlissToken.findall(s):
	    if token[0] != '<':
		self.addOrth(token)


class LexiconExtractor(BlissCorpusParser):
    def addOrthography(self, orth):
	if orth not in self.specialVocab:
	    self.vocab.add(orth)

    def __init__(self, htkLatticeSuffix, htkLatticeEncoding):
	BlissCorpusParser.__init__(self)
	self.parseSpeakerAndCondition(False)
	self.htkLatticeSuffix = htkLatticeSuffix
	self.htkLatticeEncoding = htkLatticeEncoding
	self.vocab = set()
	self.specialVocab = set()
	self.createSpecial()
	self.htkExtractor = HtkLatticeOrthographyExtractor(self.addOrthography)
	self.blissExtractor = BlissOrthographyExtractor(self.addOrthography)
	self.latticeArchiveDir = None
	self.corpusDir = None
	self.recordingDir = None
	self.segmentCounter = 0
	self.isLog = False

    def setLog(self, isLog):
	self.isLog = isLog

    def createSpecial(self):
	self.special = {
	    'silence'        : ( ('[silence]', '[SILENCE]'), None),
	    'unknown'        : ( ('<unk>', '<UNK>', '[unknown]', '[UNKNOWN]'), '[UNKNOWN]'),
	    '???'            : ( ('[???]',), None),
	    'sentence-begin' : ( ('<s>', '<S>', '[SENTENCE-BEGIN]'), None),
	    'sentence-end'   : ( ('</s>', '</S>', '[SENTENCE-END]'), None) }
	for key, value in self.special.iteritems():
	    for orth in value[0]:
		self.specialVocab.add(orth)

    def startCorpus(self, attr):
	self.corpusDir = os.path.join(self.latticeArchiveDir, attr.get('name', ''))

    def startRecording(self, attr):
	self.recordingDir = os.path.join(self.corpusDir, attr.get('name', ''))
	self.segmentCounter = 1

    def startSegment(self, attr):
	htkLatticePath = os.path.join(self.recordingDir, attr.get('name', str(self.segmentCounter)) + self.htkLatticeSuffix)
	self.segmentCounter += 1
	if os.path.exists(htkLatticePath):
	    fd = uopen(htkLatticePath, self.htkLatticeEncoding, 'r')
	    self.htkExtractor.extract(fd)
	    uclose(fd)
	    if self.isLog:
		print >> sys.stderr, htkLatticePath, '-->'
	else:
	    print >> sys.stderr, 'Warning:', htkLatticePath, 'does not exist'

    def orthElement(self, attr, content):
	self.blissExtractor.extract(content)

    def printVocab(self, lexiconPath):
	xml = openXml(lexiconPath, 'utf-8')
	xml.open('lexicon')
	xml.empty('phoneme-inventory')
	xml.comment('special lemmas')
	for special, orthEval in self.special.iteritems():
	    xml.open('lemma', special=special)
	    for orth in orthEval[0]:
		xml.element('orth', orth)
	    xml.empty('phon')
	    if orthEval[1] is None:
		xml.empty('eval')
	    else:
		xml.open('eval')
		xml.element('tok', orthEval[1])
		xml.close()
	    xml.close()
	xml.comment('regular lemmas')
	for orth in self.vocab:
	    xml.open('lemma')
	    xml.element('orth', orth)
	    xml.empty('phon')
	    xml.close()
	xml.close()
	closeXml(xml)
	if self.isLog:
	    print >> sys.stderr, '-->', lexiconPath

    def parse(self, latticeArchiveDir, corpusPath, lexiconPath):
	self.latticeArchiveDir = latticeArchiveDir
	BlissCorpusParser.parse(self, corpusPath)
	self.printVocab(lexiconPath)

if __name__ == '__main__':
    from optparse import OptionParser
    optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <bliss-corpus>'
    )
    optparser.add_option("-a", "--archive-dir", dest="latticeArchiveDir", default="",
			 help="htk slf lattice archive root; default is ./", metavar="DIR")
    optparser.add_option("-s", "--suffix", dest="suffix", default=".lat.gz",
			 help="htk slf lattice suffix; default is '.lat.gz'", metavar="SUFFIX")
    optparser.add_option("-o", "--output", dest="output", default="-",
			 help="write lexicon to FILE; default is stdout", metavar="FILE")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
			 help="produce verbose output")
    optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
			 help="encoding used for reading htk slf lattices; default is 'ascii'", metavar="ENCODING")

    if len(sys.argv) == 1:
	optparser.print_help()
	sys.exit(0)
    options, args = optparser.parse_args()
    extractor = LexiconExtractor(options.suffix, options.encoding)
    extractor.setLog(options.verbose)
    extractor.parse(options.latticeArchiveDir, args[0], options.output)
