# -*- coding: iso-8859-1 -*-

import os.path
import re
import sys
import time
from xml import sax
from xml.sax import saxutils
from blissCorpusLib import BlissOrthographyConverter, SyncSegmenter, PunctationSegmenter
from miscLib import zopen, zclose, uopen, uclose

class TrsFileParser(sax.handler.ContentHandler):
    def __init__(self):
	# controlled by corpus parser
	self.corpusParser = None
	self.segmenter = SyncSegmenter()
	self.converter = BlissOrthographyConverter()
	# handler
	self.cdata = ''
	self.trsFileName, self.audioFileName = None, None
	self.trsSegments = []
	self.startElement = self.rootStartElement
	self.endElement   = self.rootEndElement
	self.characters   = self.collectCdataAsLatin1
	self.spkIdDict = {}
	# parser
	self.parser = sax.make_parser()
	self.parser.setFeature(sax.handler.feature_namespaces, False)
	self.parser.setFeature(sax.handler.feature_external_ges, False)
	self.parser.setFeature(sax.handler.feature_external_pes, False)
	self.parser.setContentHandler(self)

    # modify behaviour
    def getRecordingName(self):
	name = os.path.basename(self.trsFileName)
	if name.lower().endswith('.gz'):
	    name = name[:-3]
	if name.lower().endswith('.trs'):
	    name = name[:-4]
	return name

    def getAudioFileName(self):
	return self.audioFileName
    def setAudioFileName(self, audioFileName):
	self.audioFileName = audioFileName
	return self.audioFileName

    # parse
    def parse(self, trsFileName = '-'):
	# add unknown speaker
	self.spkIdDict['unk'] =  self.corpusParser.speaker({ 'id': 'unk', 'type': 'unknown', 'gender': 'unknown' })

	self.trsFileName = trsFileName
	trsFd  = zopen(self.trsFileName, 'r')
	self.parser.parse(trsFd)
	zclose(trsFd)
	print trsFileName, '-->'

    # here starts the parsing ...
    # Root
    def rootStartElement(self, name, attr):

	if name == 'Trans':
	    if self.audioFileName == None:
		self.audioFileName = attr['audio_filename']
	    self.startElement = self.transStartElement
	    self.endElement   = self.transEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Root:', name


    def rootEndElement(self, name):
	print >> sys.stderr, 'Unexpected Tag in Root:', name


    # Trans
    def transStartElement(self, name, attr):
	if name == 'Topics':
	    self.startElement = self.topicStartElement
	    self.endElement   = self.topicEndElement
	elif name == 'Speakers':
	    self.startElement = self.speakerStartElement
	    self.endElement   = self.speakerEndElement
	elif name == 'Episode':
	    self.startElement = self.episodeStartElement
	    self.endElement   = self.episodeEndElement
	    # self.corpusParser.recording(self.audioFileName)
	    self.corpusParser.recording(self.getRecordingName(), self.getAudioFileName() )
	else:
	    print >> sys.stderr, 'Unexpected Tag in Trans:', name


    def transEndElement(self, name):
	if name == 'Trans':
	    self.startElement = self.rootStartElement
	    self.endElement   = self.rootEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Trans:', name


    # Topic
    def topicStartElement(self, name, attr):
	if name == 'Topic':
	    pass
	else:
	    print >> sys.stderr, 'Unexpected Tag in Topics:', name
	pass


    def topicEndElement(self, name):
	if name == 'Topic':
	    pass
	elif name == 'Topics':
	    self.startElement = self.transStartElement
	    self.endElement   = self.transEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Topics:', name


    # Speaker
    def speakerStartElement(self, name, attr):
	if name == 'Speaker':
	    # attrDict = dict( [ (k, v) \
	    #                    for k, v in attr.items()                       \
	    #                    if not k == 'id'] )                            \
	    id, attrDict = 'unknown', {}
	    for k, v in attr.items():
		if k == 'id':
		    id = v
		elif k == 'type':
		    attrDict['gender'] = v
		elif v:
		    attrDict[k] = v
	    spkId = self.corpusParser.speaker(attrDict)
	    self.spkIdDict[id] = spkId
	else:
	    print >> sys.stderr, 'Unexpected Tag in Topics:', name


    def speakerEndElement(self, name):
	if name == 'Speaker':
	    pass
	elif name == 'Speakers':
	    self.startElement = self.transStartElement
	    self.endElement   = self.transEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Topics:', name


    # Episode
    def episodeStartElement(self, name, attr):
	if name == 'Section':
	    self.condType = attr.get('type', 'unknown')
	    self.startElement = self.sectionStartElement
	    self.endElement   = self.sectionEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Episode:', name


    def episodeEndElement(self, name):
	if name == 'Episode':
	    self.startElement = self.transStartElement
	    self.endElement   = self.transEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Episode:', name


    # Section
    def sectionStartElement(self, name, attr):
	if name == 'Turn':
	    speakerId = attr.get('speaker', 'unk')
	    if ' ' in speakerId:
		self.speakerIdList = [ self.spkIdDict[id] for id in speakerId.split() ]
		speakerId = self.speakerIdList[0]
	    else:
		speakerId  = self.spkIdDict[speakerId]
	    self.corpusParser.speakerId(speakerId)
	    self.condId    = self.corpusParser.condition( self.getCondDict( attr ) )
	    self.startElement = self.turnStartElement
	    self.endElement   = self.turnEndElement
	    self.syncStartTime = float(attr['startTime'])
	    self.turnEndTime   = float(attr['endTime'])
	    self.cdata = ''
	else:
	    print >> sys.stderr, 'Unexpected Tag in Section:', name

    def getCondDict(self, attr):
	return { 'type'		: self.condType,                   \
		 'mode' 	: attr.get('mode',     'unknown'), \
		 'fidelity'    	: attr.get('fidelity', 'unknown'), \
		 'environment' 	: attr.get('channel' , 'unknown') }

    def sectionEndElement(self, name):
	if name == 'Section':
	    self.startElement = self.episodeStartElement
	    self.endElement   = self.episodeEndElement
	else:
	    print >> sys.stderr, 'Unexpected Tag in Section:', name


    # Turn
    # write bliss-segments
    def dumpTrsSegments(self):
	if self.trsSegments:
	    blissSegments = self.segmenter.getSegmentList(self.trsSegments)
	    for start, end, orth in blissSegments:
		try:
		    self.corpusParser.segment(start, end, self.converter.getOrthography(orth))
		except:
		    print >> sys.stderr, 'File      :', self.trsFileName
		    print >> sys.stderr, 'Segment   :', time.strftime('%H:%M:%S',time.gmtime(start)), '-', time.strftime('%H:%M:%S',time.gmtime(end)), [ orth ]
		    print >> sys.stderr, 'Exception :', sys.exc_info()[0]
		    print >> sys.stderr
	    self.trsSegments = []

    def appendTrsSegment(self, time):
	if time > self.syncStartTime:
	    self.trsSegments.append( (self.syncStartTime, time, self.cdata) )
	    self.cdata = ''

#    def skipUntilSync(self):
#	self.startElement = self.turnSkipUntilSyncStartElement
#	self.endElement   = self.turnSkipUntilSyncEndElement


    # -> main
    def turnStartElement(self, name, attr):
	if name == 'Sync':
	    time = float(attr['time'])
	    self.appendTrsSegment(time)
	    self.syncStartTime = time
	elif name == 'Who':
	    if self.cdata.strip() == '':
		self.dumpTrsSegments()
		self.corpusParser.speakerId(self.speakerIdList[int(attr['nb']) - 1])
	    else:
		# print >> sys.stderr, 'WARNING(' + str(self.syncStartTime) + 's):', \
		#      'Who-Tag without preceeding Sync-Tag (i.e. overlapping speech), skip until next sync'
		# self.skipUntilSync()
		#
		# The problem occuring here is that a apeaker change occured and
		# no timestamp is given, i.e. it is not known WHEN the change occured.
		# Two possibe solutions are
		# 1) just skip everything until the next timestamp,
		#    i.e. call self.skipUntilSync()
		# 2) try to represent the facts somehow in bliss
		# Because there seems to be no "Stein der Weisen"
		# and the best solution is application dependent, it is left
		# to the user to define an appropriate handling
		raise Exception("ERROR: explicit handling needed")
	else:
	    self.cdata += '\n<' + name + ' ' \
		+ ' '.join([ k + '="' + v + '"' for k, v in attr.items() ]) \
		+ '>'


    def turnEndElement(self, name):
	if name == 'Sync':
	    pass
	elif name == 'Background':
	    pass
	elif name == 'Who':
	    pass
	elif name == 'Turn':
	    self.appendTrsSegment(self.turnEndTime)
	    self.dumpTrsSegments()
	    self.startElement = self.sectionStartElement
	    self.endElement   = self.sectionEndElement
	else:
	    self.cdata += ' </' + name + '>'


    # -> skip until next sync
##    def turnSkipUntilSyncStartElement(self, name, attr):
##	if name == 'Sync':
##	    self.startElement = self.turnStartElement
##	    self.endElement   = self.turnEndElement
##	    self.cdata = ''
##	    self.syncStartTime = float(attr['time'])
##	elif name == 'Who':
##	    self.speakerId = self.speakerIdList[int(attr['nb'])]
##	else:
##	    pass

##    def turnSkipUntilSyncEndElement(self, name):
##	if name == 'Turn':
##	    self.dumpTrsSegments()
##	    self.startElement = self.sectionStartElement
##	    self.endElement   = self.sectionEndElement
##	else:
##	    pass


    # -> skip turn
##    def turnSkipStartElement(self, name, attr):
##	pass

##    def turnSkipEndElement(self, name):
##	if name == 'Turn':
##	    self.startElement = self.sectionStartElement
##	    self.endElement   = self.sectionEndElement
##	else:
##	    pass


    # collect cdata
    def collectCdataAsLatin1(self, cdata):
	self.cdata += cdata



# ******************************************************************************


class EventConverter(BlissOrthographyConverter):
    eventRE     = re.compile(r'<Event([^>]*)>\s*</Event>')
    attrSplitRE = re.compile(r'(\w+)="(.*?)"')
    def convertEvents(self, s):
	return self.eventRE.sub(lambda m: self.eventHandler( dict( self.attrSplitRE.findall(m.group(1))) ), s)

    def eventHandler(self, attr):
	raise NotImplemented


class EventParser(BlissOrthographyConverter):
    eventRE     = re.compile(r'<Event([^>]*)>')
    attrSplitRE = re.compile(r'(\w+)="(.*?)"')

    def parse(self, s):
	eventQueue, nextEventQueue = [], []
	tokenList = []
	isEvent = False
	isComment = False
	attr = {}
	for token in self.split(s):
	    token = token.strip()
	    if token.startswith('<Event'):
		attr = dict( self.attrSplitRE.findall( self.eventRE.match(token).group(1) ) )
		extend, type, desc = attr['extent'], attr['type'], attr['desc']

		if extend == "instantaneous":
		    tokenList.append( (1, (type, desc)) )
		elif extend == "begin":
		    tokenList.append( (2, (type, desc)) )
		    eventQueue.append( (type, desc) )
		elif extend == "end":
		    if eventQueue:
			beginType, beginDesc = eventQueue.pop()
			assert type == beginType and desc == beginDesc
		    else:
			tokenList.insert(0, (2, (type, desc)) )
		    tokenList.append( (3, (type, desc)) )
		elif extend == "previous":
		    stack = []
		    try:
			while tokenList[-1][0] != 0:
			    stack.append(tokenList.pop())
		    except:
			err = 'event extent is "previous", but there is no previous token'
			raise err
		    else:
			stack.append( (3, (type, desc)) )
			stack.append(tokenList.pop())
			tokenList.append( (2, (type, desc)) )
			stack.reverse()
			tokenList.extend(stack)
		elif extend == "next":
		    nextEventQueue.append( (type, desc) )
		else:
		    err = 'unknown extent: "' + extend + '"'
		    raise Exception(err)

		isEvent = True
	    elif token == '</Event>':
		isEvent = False
	    elif token.startswith('<Comment'):
		isComment = True
	    elif token.startswith('</Comment'):
		isComment = False
	    elif token.startswith('<'):
		err = 'tag "' + token + '" found but not expected'
		raise Exception(err)
	    elif token.startswith('['):
		err = '"' + token + '" found but not expected (confused by leading "[")'
		raise Exception(err)
	    elif token and not isComment:
		if isEvent:
		    raise Exception('event-tag is expected to be empty')
		for event in nextEventQueue:
		    tokenList.append( (2, event) )
		tokenList.append( (0, (token,)) )
		while nextEventQueue:
		    tokenList.append( (3, nextEventQueue.pop()) )
	for type, desc in eventQueue:
	    tokenList.append( (3, (type, desc)) )

	handler = [
	    self.orthHandler,
	    self.instantaneousEvent,
	    self.beginEvent,
	    self.endEvent
	    ]

	for i, event in tokenList:
	    handler[i](*event)


    def instantaneousEvent(self, type, desc):
	raise NotImplemented

    def beginEvent(self, type, desc):
	raise NotImplemented

    def endEvent(self, type, desc):
	raise NotImplemented

    def orthHandler(self, token):
	raise NotImplemented



# example class using the EventParser class to
# transform transcribers crude, pseudo xml-fromat
# into nice, clean xml
class EventNormalizer(EventParser):
    def instantaneousEvent(self, type, desc):
	self.token.append('<event type="' + type + '" desc="' + desc + '"/>')

    def beginEvent(self, type, desc):
	self.token.append('<event type="' + type + '" desc="' + desc + '">')

    def endEvent(self, type, desc):
	self.token.append('</event>')

    def orthHandler(self, token):
	self.token.append(token)

    def getOrthography(self, s):
	self.token = []
	self.parse(s)
	return ' '.join(self.token)
