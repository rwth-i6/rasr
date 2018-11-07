#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

if __name__ == '__main__':
    import os, sys
    binDir = sys.argv[0]
    while os.path.islink(binDir):
	binDir = os.path.join(os.path.dirname(binDir), os.readlink(binDir))
    binDir = os.path.dirname(binDir)
    sys.path = [binDir, os.path.join(os.path.join(binDir, '..'), 'lib')] + sys.path

import sys
from ioLib import uopen, uclose
from xmlWriterLib import openXml, closeXml


class StmToBlissConverter:
    def __init__(self, corpusName='unnamed'):
	self.corpusName_ = corpusName
	self.isLog = False

    def setLog(self, isLog):
	self.isLog = isLog

    def addCategory_(self, i, name):
	while len(self.category_) <= i:
	    self.category_.append(None)
	self.category_[i] = ( (name, {}) )

    def processHeaderLine_(self, line):
	line = line.strip()
	if line.startswith(';;'):
	    line = line[2:].lstrip().lower()
	    if line.startswith('category'):
		token = line.split('"')
		i, name, description = int(token[1]), token[3], token[5]
		self.addCategory_(i, name)
		self.currentCategory_ = self.category_[i]
	    elif line.startswith('label'):
		token = line.split('"')
		key, value, desc = token[1], token[3], token[5]
		self.currentCategory_[1][key] = value
	else:
	    self.segments_ = []
	    # assume category "0" to define split (yet not considered)
	    if len(self.category_) >= 2 and self.category_[1] is not None:
		self.splits_ = self.category_[0][1]
	    else:
		self.splits_ = {}
	    # assume category "1" to be condition
	    self.environments_ = {}
	    if len(self.category_) >= 2 and self.category_[1] is not None:
		if self.category_[1][0].find('condition') == -1:
		    print >> sys.stderr, 'Warning:', 'Does category 1 \"' + self.category_[1][0] +'\" really define conditions?'
		for index, desc in self.category_[1][1].iteritems():
		    self.environments_[index] = index + '(' + desc + ')'
	    self.nCond_, self.conditions_ = 0, {}
	    # assume category "2" to be gender
	    if len(self.category_) >= 3 and self.category_[2] is not None:
		if self.category_[2][0].find('speaker sex') == -1:
		    print >> sys.stderr, 'Warning:', 'Does category 2 \"' + self.category_[2][0] +'\" really define genders?'
		self.genders_ = self.category_[2][1]
	    else:
		self.genders_ = {}
	    del self.currentCategory_
	    self.processLine_ = self.processBodyLine_
	    self.nSpk_, self.speakers_ = 0, {}


    def processBodyLine_(self, line):
	i = line.find(';;')
	if i != -1:
	    line = line[:i].strip()
	else:
	    line = line.strip()
	if line:
	    token = line.split(None, 6)
	    assert len(token) >= 6
	    name = token[2].lower()
	    if name != 'inter_segment_gap' and name != 'excluded_region':
		if len(token) == 6:
		    print >> sys.stderr, 'Warning: Empty segment \"' + line + '\"'
		    token.append('')
		self.processSegment_(token[0], token[1], token[2], token[3], token[4], self.parseTags_(token[5]), token[6])

    def parseTags_(self, s):
	tags = s[1:-1].lower().split(',')
	if len(tags) < 3:
	    if len(tags) < 2:
		if len(tags) < 1:
		    tags.append('0')
		tags.append('unknown')
	    tags.append('unknown')
	return tags

    def getConditionId_(self, channel, env):
	cond = (channel, env)
	condId = self.conditions_.setdefault(cond, str(self.nCond_))
	self.nCond_ = len(self.conditions_)
	return condId

    def getSpeakerId_(self, name, gender):
	spk = (name, gender)
	spkId = self.speakers_.setdefault(spk, str(self.nSpk_))
	self.nSpk_ = len(self.speakers_)
	return spkId

    def processSegment_(self, recName, channel, speakerName, startTime, endTime, tags, ref):
	condId = self.getConditionId_(channel, self.environments_.get(tags[1], tags[1]))
	spkId = self.getSpeakerId_(speakerName, self.genders_.get(tags[2], tags[2]))
	self.segments_.append( (recName, startTime, endTime, condId, spkId, ' '.join(ref.split())) )

    def writeBliss_(self, xml):
	conditions = [ (v, k) for k, v in self.conditions_.iteritems() ]
	conditions.sort()
	speakers = [ (v, k) for k, v in self.speakers_.iteritems() ]
	speakers.sort()

	xml.open('corpus', name=self.corpusName_)
	for condId, cond in conditions:
	    xml.open('condition-description', name=condId)
	    xml.element('channel', cond[0])
	    xml.element('environment', cond[1])
	    xml.close()
	for spkId, spk in speakers:
	    xml.open('speaker-description', name=spkId)
	    xml.element('name', spk[0])
	    xml.element('gender', spk[1])
	    xml.close()
	currentRecName = None
	for recName, startTime, endTime, condId, spkId, ref in self.segments_:
	    if recName != currentRecName:
		if currentRecName is not None:
		    xml.close()
		xml.open('recording', name=recName, audio=recName)
		currentRecName = recName
	    segName = recName+'-'+startTime+'-'+endTime
	    xml.open('segment', name=segName, start=startTime, end=endTime)
	    xml.empty('condition', name=condId)
	    xml.empty('speaker', name=spkId)
	    xml.element('orth', ref)
	    xml.close()
	xml.close()
	xml.close()

    def parse(self, pathIn, pathOut, encoding):
	self.category_ = []
	self.processLine_ = self.processHeaderLine_
	fd = uopen(pathIn, encoding, 'r')
	for line in fd:
	    self.processLine_(line)
	uclose(fd)
	xml = openXml(pathOut, encoding)
	self.writeBliss_(xml)
	closeXml(xml)
	if self.isLog:
	    print >> sys.stderr, pathIn, '-->', pathOut



if __name__ == '__main__':
    from optparse import OptionParser
    optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <stm-file>'
    )
    optparser.add_option("-o", "--output", dest="output", default="-",
			 help="write filtered lexicon to FILE; default is stdout", metavar="FILE")
    optparser.add_option("-t", "--corpus-name", dest="corpusName", default="unnamed",
			 help="name of bliss corpus", metavar="STRING")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
			 help="produce verbose output")
    optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
			 help="encoding used for reading the glm file; default is 'ascii'", metavar="ENCODING")

    if len(sys.argv) == 1:
	optparser.print_help()
	sys.exit(0)
    options, args = optparser.parse_args()
    converter = StmToBlissConverter(options.corpusName)
    converter.setLog(options.verbose)
    converter.parse(args[0], options.output, options.encoding)
