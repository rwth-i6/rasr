#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os.path
import sys

if __name__ == '__main__':
    import os, sys
    binDir = sys.argv[0]
    while os.path.islink(binDir):
	binDir = os.path.join(os.path.dirname(binDir), os.readlink(binDir))
    binDir = os.path.dirname(binDir)
    sys.path = [binDir, os.path.join(os.path.join(binDir, '..'), 'lib')] + sys.path

from xml import sax
from ioLib import zopen, zclose
from xmlWriterLib import openXml, closeXml
from xmlParserLib import SimpleXmlParser

INFINITY = float('inf')

class ErrorStatistic:
    __slots__ = ('nSeg', 'nFalseSeg', 'nTok', 'nDel', 'nIns', 'nSub', 'sumDensity')

    def __init__(self):
	self.nSeg = self.nFalseSeg = self.nTok = self.nDel = self.nIns = self.nSub = 0
	self.sumDensity = 0.0

    def wer(self):
	if self.nTok == 0:
	    return float('nan')
	else:
	    return float(self.nDel + self.nIns + self.nSub) / float(self.nTok)

    def writeXml(self, xml):
	if self.nSeg == 0:
	    ser = float('nan')
	    density = 0.0
	else:
	    ser = float(self.nFalseSeg) / float(self.nSeg)
	    density = self.sumDensity / float(self.nSeg)
	if self.nTok == 0:
	    wer = float('nan')
	else:
	    wer = float(self.nDel + self.nIns + self.nSub) / float(self.nTok)
	if density > 0.0:
	    xml.element('average-lattice-density', str(density))
	xml.open('sentence-error-rate')
	xml.element('event', str(self.nSeg), type='segment')
	xml.element('SER', str(ser))
	xml.close()
	xml.open('word-error-rate')
	xml.element('event', str(self.nTok), type='token')
	xml.element('event', str(self.nDel), type='deletion')
	xml.element('event', str(self.nIns), type='insertion')
	xml.element('event', str(self.nSub), type='substitution')
	xml.element('WER', str(wer))
	xml.close()


class ErrorStatisticAccumulator:
    def __init__(self):
	self.stats_ = []

    def density(self, density):
	for stat in self.stats_:
	    stat.sumDensity += density

    def incSeg(self):
	for stat in self.stats_:
	    stat.nSeg += 1

    def incFalseSeg(self):
	for stat in self.stats_:
	    stat.nFalseSeg += 1

    def addTok(self, n):
	for stat in self.stats_:
	    stat.nTok += n

    def addDel(self, n):
	for stat in self.stats_:
	    stat.nDel += n

    def addIns(self, n):
	for stat in self.stats_:
	    stat.nIns += n

    def addSub(self, n):
	for stat in self.stats_:
	    stat.nSub += n


class ErrorStatisticCollection:
    class NamedStatistic:
	def __init__(self, name):
	    self.name = name
	    self.all = ErrorStatistic();
	    self.speakers = {}
	    self.conditions = {}

    def __init__(self, bySpeaker, byCondition):
	self.stats_ = {}
	self.bySpeaker = bySpeaker
	self.speakerDb = {}
	self.byCondition = byCondition
	self.conditionDb = {}

    def getSpeakerIds(self):
	spkIds = []
	for stat in self.stats_.itervalues():
	    spkIds += stat.speakers.keys()
	return set(spkIds)

    def getConditionIds(self):
	condIds = []
	for stat in self.stats_.itervalues():
	    condIds += stat.conditions.keys()
	return set(condIds)

    def setSpeakerDb(self, speakerDb):
	self.speakerDb = speakerDb

    def setConditionDb(self, conditionDb):
	self.conditionDb = conditionDb

    def get(self, name, speakerId, condId):
	stat = self.stats_.setdefault(name, ErrorStatisticCollection.NamedStatistic(name))
	accu = ErrorStatisticAccumulator()
	accu.stats_.append(stat.all)
	if self.bySpeaker and speakerId:
	    accu.stats_.append(stat.speakers.setdefault(speakerId, ErrorStatistic()))
	if self.byCondition and condId:
	    accu.stats_.append(stat.conditions.setdefault(condId, ErrorStatistic()))
	return accu

    def wer(self):
	statList = self.stats_.items()
	statList.sort()
	for statName, stat in statList:
	    yield (statName, stat.all.wer())

    def writeXml(self, xml):
	statList = self.stats_.items()
	statList.sort()
	xml.open('report')
	for statName, stat in statList:
	    xml.open('statistics', name=statName)
	    if self.byCondition:
		conditionList = stat.conditions.items()
		conditionList.sort()
		for condId, condStat in conditionList:
		    condProps = self.conditionDb.get(condId)
		    if condProps:
			xml.open('condition', id=condId, environment=condProps.get('environment', 'unknown'), channel=condProps.get('channel', 'unknown'))
		    else:
			xml.open('condition', id=condId)
		    condStat.writeXml(xml)
		    xml.close()
	    if self.bySpeaker:
		speakerList = stat.speakers.items()
		speakerList.sort()
		for spkId, spkStat in speakerList:
		    spkProps = self.speakerDb.get(spkId)
		    if spkProps:
			xml.open('speaker', id=spkId, name=spkProps.get('name', 'unknown'), gender=spkProps.get('gender', 'unknown'))
		    else:
			xml.open('speaker', id=spkId)
		    spkStat.writeXml(xml)
		    xml.close()
	    xml.open('overall')
	    stat.all.writeXml(xml)
	    xml.close()
	    xml.close()
	xml.close()


class ErrorStatisticCollectionAccumulator(SimpleXmlParser):
    def __init__(self):
	SimpleXmlParser.__init__(self)
	self.bySpeaker = False
	self.byCondition = False

    def groupBySpeaker(self, b = True):
	self.bySpeaker = b

    def groupByCondition(self, b = True):
	self.byCondition = b

    def reset(self):
	self.stats_ = ErrorStatisticCollection(self.bySpeaker, self.byCondition)
	self.startElement = self.startBoondocks
	self.endElement   = self.endBoondocks
	self.processCdata = self.ignoreCdata
	self.cdata_ = ''
	self.encoding = ''
	# ...
	self.condId_  = None
	self.spkId_   = None
	self.typeId_ = ''
	self.layerId_ = ''
	self.evalId_ = ''
	self.accu_ = None
	self.err_ = 0

    def id(self):
	_id = self.evalId_
	if self.layerId_:
	    _id =  self.layerId_ + '/' + _id
	if self.typeId_:
	    _id = self.typeId_ + '/' + _id
	return _id

    def startFile(self, path, encoding):
	self.encoding = encoding

    def startBoondocks(self, name, attr):
	if name == 'condition':
	    self.condId_ = attr.get('name', None)
	elif name == 'speaker':
	    self.spkId_ = attr.get('name', None)
	elif name == 'layer':
	    self.layerId_ = attr.get('name', None)
	    assert self.layerId_
	elif name == 'evaluation':
	    self.typeId_ = attr.get('type', None)
	    self.evalId_ = attr.get('name', 'unknown')
	    self.accu_ = self.stats_.get(self.id(), self.spkId_, self.condId_)
	    self.startElement = self.startEvaluation
	    self.endElement   = self.endEvaluation
	elif name == "word-lattice-density":
	    self.processCdata = self.collectCdata

    def endBoondocks(self, name):
	if name == 'layer':
	    self.layerId_ = None
	elif name == 'segment':
	    self.condId_  = None
	    self.spkId_   = None
	elif name == "word-lattice-density":
	    density = float(self.cdata_)
	    if density != INFINITY:
		self.accu_.density(density)
	    self.cdata_ = ''
	    self.processCdata = self.ignoreCdata

    def startEvaluation(self, name, attr):
	if name == 'statistic' and attr.get('type') == 'edit-distance':
	    self.startElement = self.startStatistic
	    self.endElement = self.endStatistic
	    self.processCdata = self.collectCdata

    def endEvaluation(self, name):
	if name == 'evaluation':
	    self.accu_.incSeg()
	    if self.err_ > 0:
		self.accu_.incFalseSeg()
		self.err_ = 0
	    self.startElement = self.startBoondocks
	    self.endElement = self.endBoondocks

    def startStatistic(self, name, attr):
	if name == 'count':
	    self.currentEvent_ = attr.get('event', None)

    def endStatistic(self, name):
	if name == 'count':
	    if self.currentEvent_ is not None:
		n = int(self.cdata_)
		if self.currentEvent_ == 'token':
		    self.accu_.addTok(n)
		elif self.currentEvent_ == 'deletion':
		    self.err_ += n
		    self.accu_.addDel(n)
		elif self.currentEvent_ == 'insertion':
		    self.err_ += n
		    self.accu_.addIns(n)
		elif self.currentEvent_ == 'substitution':
		    self.err_ += n
		    self.accu_.addSub(n)
	    self.currentEvent_ = None
	    self.cdata_ = ''
	elif name == 'statistic':
	    self.startElement = self.startEvaluation
	    self.endElement = self.endEvaluation
	    self.processCdata = self.ignoreCdata

    def ignoreCdata(self, cdata):
	pass

    def collectCdata(self, cdata):
	self.cdata_ += cdata

    def characters(self, cdata):
	self.processCdata(cdata)

    def accumulate(self, path):
	self.reset()
	SimpleXmlParser.parse(self, path)
	return self.stats_



class SpeakerAndConditionExtractor(sax.handler.ContentHandler):
    def __init__(self, spkIds, condIds):
	self.spkIds_ = spkIds
	self.condIds_ = condIds
##	self.spk_ = dict( ( (spkId, {}) for spkId in self.spkIds_ ) )
##	self.cond_ = dict( ( (condId, {}) for condId in self.condIds_ ) )
	self.spk_ = dict( [ (spkId, {}) for spkId in self.spkIds_ ] )
	self.cond_ = dict( [ (condId, {}) for condId in self.condIs_ ] )
	self.nMissing_ = len(self.spkIds_) + len(self.condIds_)

	self.processCdata = self.ignoreCdata
	self.currentCond_ = None
	self.currentSpk_ = None
	self.cdata_ = ''

    def missing(self):
	return self.nMissing_

    def getSpeakerDb(self):
	return self.spk_

    def getConditionDb(self):
	return self.cond_

    def ignoreCdata(self, cdata):
	pass

    def collectCdata(self, cdata):
	self.cdata_ += cdata

    def characters(self, cdata):
	self.processCdata(cdata)

    def startElement(self, name, attr):
	if name == "include":
	    self.parseInclude(attr['file'])
	elif name == 'condition-description':
	    self.currentCond_ = self.cond_.get(attr['name'], None)
	    if self.currentCond_ is not None:
		self.processCdata = self.collectCdata
	elif name == 'speaker-description':
	    self.currentSpk_ = self.spk_.get(attr['name'], None)
	    if self.currentSpk_ is not None:
		self.processCdata = self.collectCdata

    def endElement(self, name):
	if self.currentCond_ is not None:
	    if name == 'condition-description':
		self.currentCond_ = None
		self.nMissing_ -= 1
		self.processCdata = self.ignoreCdata
	    else:
		self.currentCond_[name] = ' '.join(self.cdata_.split())
	    self.cdata_ = ''
	elif self.currentSpk_ is not None:
	    if name == 'speaker-description':
		self.currentSpk_ = None
		self.nMissing_ -= 1
		self.processCdata = self.ignoreCdata
	    else:
		self.currentSpk_[name] = ' '.join(self.cdata_.split())
	    self.cdata_ = ''

    def parseInclude(self, path):
	self.parseFile(os.path.join(self.base, path))

    def parseFile(self, path) :
	parser = sax.make_parser()
	parser.setFeature(sax.handler.feature_namespaces, 0)
	parser.setFeature(sax.handler.feature_external_ges, False)
	parser.setFeature(sax.handler.feature_external_pes, False)
	parser.setContentHandler(self)
	fd = zopen(path, 'r')
	for line in fd:
	    if self.nMissing_ == 0:
		break
	    else:
		parser.feed(line)
	zclose(fd)

    def parse(self, path):
	self.base = os.path.dirname(path)
	self.parseFile(path)


def calculateErrorStatistics(logFile, bySpeaker = False, byCondition = False, corpusPath = None):
    accu = ErrorStatisticCollectionAccumulator()
    accu.groupBySpeaker(bySpeaker)
    accu.groupByCondition(byCondition)
    stats = accu.accumulate(logFile)
    if corpusPath and (bySpeaker or byCondition):
	spkAndCond = SpeakerAndConditionExtractor(stats.getSpeakerIds(), stats.getConditionIds())
	spkAndCond.parse(corpusPath)
	stats.setSpeakerDb(spkAndCond.getSpeakerDb())
	stats.setConditionDb(spkAndCond.getConditionDb())
    return accu, stats


def wer(logFile):
    accu = ErrorStatisticCollectionAccumulator()
    accu.groupBySpeaker(False)
    accu.groupByCondition(False)
    stats = accu.accumulate(logFile)
    return stats.wer()


if __name__ == '__main__':
    from optparse import OptionParser
    optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <result-file>'
    )
    optparser.add_option("-o", "--output", dest="output", default="-",
			 help="write statistic to FILE; default is stdout", metavar="FILE")
    optparser.add_option("-s", "--speaker", dest="bySpeaker", action="store_true", default=False,
			 help= "dump statistics by speaker")
    optparser.add_option("-c", "--condition", dest="byCondition", action="store_true", default=False,
			 help= "dump statistics by condition")
    optparser.add_option("-b", "--corpus", dest="corpusPath", default=None,
			 help= "bliss corpus; used to get speaker and condition names", metavar="FILE")

    if len(sys.argv) == 1:
	optparser.print_help()
	sys.exit(0)
    options, args = optparser.parse_args()
    accu, stats = calculateErrorStatistics(args[0], options.bySpeaker, options.byCondition, options.corpusPath)
    xml = openXml(options.output, accu.encoding)
    stats.writeXml(xml)
    closeXml(xml)
