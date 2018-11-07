__version__   = '$Revision$'
__date__      = '$Date$'

from xmlparser import *
import os

class NamedCorpusEntity(object):
    def __init__(self):
	self.parent = None
	self.name = None

    def setName(self, name):
	self.name = name

    def getFullName(self):
	if self.parent:
	    return self.parent.fullName + '/' + self.name
	else:
	    return self.name
    fullName = property(getFullName)

class Speaker(NamedCorpusEntity):
    pass

class AcousticCondition(NamedCorpusEntity):
    pass

class ParentEntity(NamedCorpusEntity):
    def __init__(self):
	super(ParentEntity, self).__init__()
	self.childrensNames = set()

    def isNameReserved(self, name):
	return name in self.childrensNames

    def reserveName(self, name):
	self.childrensNames.add(name)

class CorpusSection(ParentEntity):
    def __init__(self, parent):
	super(CorpusSection, self).__init__()
	self.parent = parent
	if parent:
	    self.level = parent.level + 1
	else:
	    self.level = 0
	self.speakers = {}
	self.conditions = {}

class Corpus(CorpusSection):
    pass

class Recording(CorpusSection):
    pass

class Segment(CorpusSection):
    orth = None
    startTime = None
    endTime = None


class CorpusVisitor(object):
    def enterCorpus(self, corpus): pass
    def leaveCorpus(self, corpus): pass
    def enterRecording(self, recording): pass
    def leaveRecording(self, recording): pass
    def visitSegment(self, segment): pass


class OrthographyElement(XmlMixedElement):
    def __init__(self, handler):
	super(OrthographyElement, self).__init__('orth')
	self.flattenUnknownElements()
	self.handler = handler

    def start(self, atts):
	self.data = []

    def characters(self, data):
	self.data.append(data)

    def end(self):
	orth = ' '.join((''.join(self.data)).split())
	self.handler(orth)


class CorpusDescriptionParser(XmlSchemaParser):
    visitor = None

    def __init__(self, description):
	super(CorpusDescriptionParser, self).__init__()
	self.initSchema()
	self.description = description
	self.isSubParser = False
	self.superCorpus = None

    def initSchema(self):
	speakerDesc = XmlIgnoreElement('speaker-description')
	conditionDesc = XmlIgnoreElement('condition-description')

	orth = OrthographyElement(self.setOrth)
	orth.flattenUnknownElements()

	segment = XmlRegularElement(
	    'segment',
	    self.startSegment, self.endSegment)
	segment.addTransition(segment.initial, segment.initial, orth)
	segment.addFinalState(segment.initial)
	segment.ignoreUnknownElements()

	recording = XmlRegularElement(
	    'recording',
	    self.startRecording, self.endRecording)
	recording.addTransition(recording.initial, recording.initial, segment)
	recording.addFinalState(recording.initial)
	recording.ignoreUnknownElements()

	include = XmlEmptyElement('include', self.include)

	# should use XmlRegularElement, but cannot due to nesting restriction
	subcorpus = XmlMixedElement(
	    "subcorpus", self.startSubcorpus, self.endSubcorpus)
	subcorpus.addChild(subcorpus)
	subcorpus.addChild(include)
	subcorpus.addChild(speakerDesc)
	subcorpus.addChild(conditionDesc)
	subcorpus.addChild(recording)
	subcorpus.ignoreUnknownElements()

	corpus = XmlRegularElement(
	    'corpus', self.startCorpus, self.endCorpus)
	corpus.addTransition(corpus.initial, corpus.initial, subcorpus)
	corpus.addTransition(corpus.initial, corpus.initial, include)
	corpus.addTransition(corpus.initial, corpus.initial, speakerDesc)
	corpus.addTransition(corpus.initial, corpus.initial, conditionDesc)
	corpus.addTransition(corpus.initial, corpus.initial, recording)
	corpus.addFinalState(corpus.initial)
	corpus.ignoreUnknownElements()

	self.setRoot(corpus)

    def includeFile(self, relativeFilename):
	filename = os.path.join(self.corpusDir, relativeFilename)
	subParser = CorpusDescriptionParser(self.description)
	subParser.isSubParser = True
	subParser.corpus = self.corpus
	subParser.accept(filename, self.visitor)

    def include(self, atts):
	corpus = atts.get('file')
	if corpus:
	    self.includeFile(corpus)
	else:
	    self.error("attribute \"file\" missing in \"include\" tag")
	    return

    def startCorpus(self, atts):
	name = atts.get('name')
	if not name:
	    error("attribute 'name' missing in 'corpus' tag")
	    return
	if self.isSubParser:
	    if name != self.corpus.name:
		warning('name mismatch on included corpus: "%s" instead of "%s"' % (
		    name, self.corpus.name))
		return
	else:
	    assert not self.superCorpus
	    self.corpus = Corpus(None)
	    self.corpus.name = name
	    if self.visitor:
		self.visitor.enterCorpus(self.corpus)
	self.currentSection = self.corpus

    def startSubcorpus(self, atts):
	name = atts.get('name')
	if not name:
	    error("attribute 'name' missing in 'subcorpus' tag")
	    return
	self.superCorpus = self.corpus
	self.corpus = Corpus(self.superCorpus)
	if self.superCorpus.isNameReserved(name):
	    error('subcorpus "%s" already defined in the section' % name)
	else:
	    self.superCorpus.reserveName(name)
	    self.corpus.name = name
	self.currentSection = self.corpus
	if self.visitor:
	    self.visitor.enterCorpus(self.corpus)

    def startRecording(self, atts):
	name = atts.get('name')
	audio = atts.get('audio')

	self.recording = Recording(self.corpus)

	if not name:
	    self.error('attribute "name" missing in "recording" tag')
	else:
	    if self.corpus.isNameReserved(name):
		self.error('recording "%s" already defined in the section' % name)
	    else:
		self.corpus.reserveName(name)
		self.recording.name = name
	if not audio:
	    self.error('attribute "audio" missing in "recording" tag')
	else:
	    audioFilename = os.path.join(self.description.audioDir, audio)
	    self.recording.audio = audioFilename
	self.currentSection = self.recording
	self.segmentNum = 0
	if self.visitor:
	    self.visitor.enterRecording(self.recording)

    def startSegment(self, atts):
	assert self.recording

	name = atts.get('name')
	self.segment = Segment(self.recording)
	self.segmentNum += 1
	if name:
	    if self.recording.isNameReserved(name):
		self.error('segment "%s" already defined in the section' % name)
	    else:
		self.recording.reserveName(name)
		self.segment.setName(name)
	else:
	    autoName = "%d" % self.segmentNum
	    while self.recording.isNameReserved(autoName):
		autoName += "+"
	    self.recording.reserveName(autoName)
	    self.segment.name = autoName
	self.segment.startTime = atts.get('start')
	self.segment.endTime = atts.get('end')

    def setOrth(self, orth):
	assert self.segment
	if self.description.shallCaptializeTranscriptions:
	    orth = orth.upper()
	self.segment.orth = orth

    def endSegment(self):
	if self.visitor:
	    self.visitor.visitSegment(self.segment)
	self.segment = None

    def endRecording(self):
	if self.visitor:
	    self.visitor.leaveRecording(self.recording)
	self.recording = None
	self.currentSection = self.corpus

    def endSubcorpus(self):
	if self.visitor:
	    self.visitor.leaveCorpus(self.corpus)
	assert self.superCorpus
	self.corpus = self.superCorpus
	self.superCorpus = self.corpus.parent
	self.currentSection = self.corpus

    def endCorpus(self):
	if not self.isSubParser:
	    assert not self.superCorpus
	    if self.visitor:
		self.visitor.leaveCorpus(self.corpus)
	    self.corpus = None
	    self.currentSection = None

    def accept(self, filename, visitor):
	self.visitor = visitor
	self.corpusDir = os.path.dirname(filename)
	self.parse(zopen(filename).read())
	self.visitor = None


from miscLib import zopen

class CorpusDescription:
    def __init__(self, filename):
	self.filename = filename
	self.audioDir = ''
	self.shallCaptializeTranscriptions = False

    def accept(self, visitor):
	parser = CorpusDescriptionParser(self)
	parser.accept(self.filename, visitor)
