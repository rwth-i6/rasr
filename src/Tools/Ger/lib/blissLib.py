# -*- coding: iso-8859-1 -*-

import os.path
from xml import sax
from xml.sax import saxutils
from ioLib import zopen, zclose

class BlissCorpusParser(sax.handler.ContentHandler):
    """
    reads a bliss corpus file and calls orthElement whenever an orth-element
    was parsed
    """
    def __init__(self):
	# handler
	self.speakerDb   = {}
	self.conditionDb = {}
	self.depth = 0
	self.content = ''
	self.base = None

	self.speaker = None
	self.condition = None

	self.attributeName = None
	self.startElement = self.xStartElement_
	self.endElement = self.xEndElement_

    # option
    def parseSpeakerAndCondition(self, isParseSpeakerAndCondition = True):
	if isParseSpeakerAndCondition:
	    self.startElement = self.xStartElement_
	    self.endElement = self.xEndElement_
	else:
	    self.startElement = self.startElement_
	    self.endElement = self.endElement_


    # interface
    def startMain(self, filename):
	pass

    def endMain(self):
	pass

    def startInclude(self, attr):
	pass

    def endInclude(self):
	pass

    def startCorpus(self, attr):
	pass

    def endCorpus(self):
	pass

    def startSubcorpus(self, attr):
	pass

    def endSubcorpus(self):
	pass

    def startRecording(self, attr):
	pass

    def endRecording(self):
	pass

    def startSegment(self, attr):
	pass

    def endSegment(self):
	pass

    def orthElement(self, attr, content):
	pass

    # handler
    def characters(self, content):
	self.content += content

    def cdata(self):
	cdata = self.content.strip()
	self.content = ''
	return cdata

    def startDocument(self):
	self.content = ''
	self.isSpeaker = False
	self.isCondition = False
	self.isOrth = False

    def endDocument(self):
	pass

    def startElement_(self, name, attr):
	if self.isOrth:
	    self.content += "<" + ' '.join([name] + map(lambda kv: '%s="%s"' % kv, attr.items())) + ">"
	else:
	    if name == "orth":
		self.orthAttr = attr
		self.isOrth = True
	    elif name == "segment":
		self.startSegment(attr)
	    elif name == "recording":
		self.startRecording(attr)
	    elif name == "corpus":
		self.depth += 1
		if self.depth == 1:
		    self.startCorpus(attr)
		else:
		    self.startSubcorpus(attr)
	    elif name == "include":
		self.startInclude(attr)
		self.parseInclude(attr['file'])
	    self.content = ''

    def endElement_(self, name):
	if self.isOrth:
	    if name == 'orth':
		self.orthElement(self.orthAttr, self.cdata())
		self.isOrth = False
	    else:
		self.content += " </" + name + ">"
	else:
	    if name == "segment":
		self.endSegment()
		self.speaker = self.condition = None
	    elif name == "recording":
		self.endRecording()
	    elif name == "corpus":
		if self.depth == 1:
		    self.endCorpus()
		else:
		    self.endSubcorpus()
		self.depth -= 1
	    elif name == "include":
		self.endInclude()
	    self.content = ''


    def xStartElement_(self, name, attr):
	if self.isSpeaker or self.isCondition:
	    self.attributeName = name
	elif name == "speaker":
	    id = attr.get('name', None)
	    self.speaker = self.speakerDb.get(id, dict(attr))
	    self.speaker['id'] = id
	elif name == "condition":
	    id = attr.get('name', None)
	    self.condition = self.conditionDb.get(id, dict(attr))
	    self.condition['id'] = id
	elif name == 'speaker-description':
	    self.isSpeaker = True
	    self.speaker = {}
	    self.speakerDb[attr['name']] = self.speaker
	elif name == 'condition-description':
	    self.isCondition = True
	    self.condition = {}
	    self.conditionDb[attr['name']] = self.condition
	else:
	    self.startElement_(name, attr)

    def xEndElement_(self, name):
	if self.isSpeaker:
	    if name == 'speaker-description':
		self.speaker = None
		self.isSpeaker = False
	    else:
		self.speaker[self.attributeName] = self.content.strip()
	    self.content = ''
	elif self.isCondition:
	    if name == 'condition-description':
		self.condition = None
		self.isCondition = False
		self.content = ''
	    else:
		self.condition[self.attributeName] = self.content.strip()
	    self.content = ''
	else:
	    self.endElement_(name)


    def parseFile(self, path) :
	parser = sax.make_parser()
	parser.setFeature(sax.handler.feature_namespaces, 0)
	parser.setFeature(sax.handler.feature_external_ges, False)
	parser.setFeature(sax.handler.feature_external_pes, False)
	parser.setContentHandler(self)
	fd = zopen(path, 'r')
	parser.parse(fd)
	zclose(fd)
	# print '-->', path

    def parseInclude(self, path):
	self.parseFile(os.path.join(self.base, path))

    def parseMain(self, path):
	self.base = os.path.dirname(path)
	self.parseFile(path)

    def parse(self, path):
	self.startMain(path)
	self.parseMain(path)
	self.endMain()
