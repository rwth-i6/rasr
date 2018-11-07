#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

from optparse import OptionParser, OptionGroup
from miscLib import uopen,uclose
import sys
import string
import re
import gzip
import os.path
import time, datetime
import math
import codecs

from xml.sax import make_parser
from xml.sax.handler import ContentHandler, feature_namespaces, feature_external_ges, feature_external_pes
from xmlWriterLib import openXml, closeXml, XmlWriter

#########################################################################
# corpus format file (cff) object: sequential table
#
# recording time-pos key data-object
# E.g.:
# 20041026_1505_1700_EN_SAT 100.0 boundary
# 20041026_1505_1700_EN_SAT 100.0 speaker {'name': 'interpreter_2_GARRIGA_POLLEDO_Salvador'}
# 20041026_1505_1700_EN_SAT 100.0 orth on an inter institutional agreement
# 20041026_1505_1700_EN_SAT 110.0 boundary
# 20041027_1000_1200_EN_SAT 20.0 boundary
# 20041027_1000_1200_EN_SAT 20.0 orth
# 20041027_1000_1200_EN_SAT 30.0 boundary
# 20041027_1000_1200_EN_SAT 50.0 boundary
# 20041027_1000_1200_EN_SAT 50.0 orth thank
# 20041027_1000_1200_EN_SAT 52.0 orth you
# 20041027_1000_1200_EN_SAT 54.0 orth very
# 20041027_1000_1200_EN_SAT 56.0 orth much
# 20041027_1000_1200_EN_SAT 58.0 boundary
#
# !!! the order of keys at the same recording and time-pos DOES NOT MATTER !!!
#
# keys            | data-object		 				|
# ----------------+---------------------------------------------------------+
# boundary        | None						|
# sync            | None						|
# orth            | string			           		|
# speaker         | dictionary, e.g.: { 'name': 'Peter', 'gender': 'male' } |
# condition	  | dictionary						|
# track           | int, i.e.: 1-N							|
# corpus-name     | list of strings, i.e.: [ 'CORPUS', 'SUBCORPUS-1', ...]	|
# score           | dictionary, e.g.: { 'confidence': 0.6, 'acoustic': 200.1 }



class Cff:
	def __init__(self):
		self.cffDict = {}

	def setRow(self, recording, timePos, key, value = None):
		if self.cffDict.has_key((str(recording), float(timePos), str(key))) \
			and self.cffDict[(str(recording), float(timePos), str(key))] != value:
			if str(key) == 'orth':
				value = self.cffDict[(str(recording), float(timePos), str(key))] +' '+ value
			else:
				print >> sys.stderr, "Warning: overwriting previous cff-rows (same key)"
		self.cffDict[(str(recording), float(timePos), str(key))] = value

	def getRows(self):
		rows = []
		keys = self.cffDict.keys()
		keys.sort()
		for key in keys:
			rows.append((key[0], key[1], key[2], self.cffDict[key]))
		return rows

	def writeRows(self, file = sys.stdout):
		for row in self.getRows():
			line = ' '.join([ row[0], str(row[1]), row[2] ])
			if row[3] != None:
				if row[2] != 'orth':
					line += ' '+repr(row[3])
				else:
					line += ' '+row[3]
			print >> file, line.encode('UTF-8')

	def readRows(self, file = sys.stdin):
		for line in file.readlines():
			line = line.decode('UTF-8')
			lineList = line[:-1].split(' ')
			value = ' '.join(lineList[3:])
			if value == '':
				self.setRow(lineList[0], lineList[1], lineList[2])
			else:
				if lineList[2] == 'orth':
					self.setRow(lineList[0], lineList[1], lineList[2], value)
				else:
					self.setRow(lineList[0], lineList[1], lineList[2], eval(value))


###########################################################
# bliss corpus parser

def blissCorpusParser(inFileName, cff = Cff()):
 parser	 = make_parser()
 handler = BlissCorpusHandler( os.path.dirname(inFileName), cff)
 parser.setContentHandler( handler )
 #parser.parse( gzip.open(inFileName, "rb") )
 parser.parse( uopen(inFileName) )
 return handler.cff

class BlissCorpusHandler(ContentHandler):
	def __init__(self, path, obj = Cff()):
		self.cff = obj
		self.readPath = path

		self.corpusNameList = []

		self.recording = None
		self.orth = None
		self.data = ""
		self.conditionId = None
		self.speakerId = None
		self.segStart = 0
		self.segEnd = None
		self.segTrack = 0

		self.descriptionId = None
		self.isDescription = False
		self.descriptionKey = None
		self.descriptionDict = None
		self.speakerDict = {}
		self.conditionDict = {}

	def startElement(self, name, attrs):
		dictAttrs = dict(attrs)

		if self.isDescription:
			self.descriptionKey = name
			self.data = ''


		if name == "include":
			self.cff = blissCorpusParser(os.path.join(self.readPath, dictAttrs["file"]), self.cff)
		elif name == "orth":
			self.orth = ''
		elif name == "recording":
			self.recording = attrs.get('audio')
			if len(self.corpusNameList) > 0:
				self.cff.setRow(self.recording, 0, 'corpus-name', self.corpusNameList)
		elif name == "segment":
			self.segStart = attrs.get('start', 0 )
			self.segEnd = attrs.get('end', None)
			self.segTrack = attrs.get('track', 0)
		elif name == 'speaker':
			self.speakerId = attrs.get('name')
		elif name == 'condition':
			self.conditionId = attrs.get('name')
		elif name == 'corpus' or name == 'subcorpus':
			self.corpusNameList.append(attrs.get('name'))
		elif name == 'speaker-description' or name == 'condition-description':
			self.descriptionId = attrs.get('name')
			self.isDescription = True
			self.descriptionDict = {}

	def endElement(self, name):
		self.data = self.data.replace("\n", " ")
		self.data = self.data.replace("\t", " ")
		self.data = re.sub(" +", " ", self.data)
		self.data = re.sub("^ ", "", self.data)
		self.data = re.sub(" $", "", self.data)

		if self.orth != None:
			self.orth += self.data+' '

		if name == "orth":
			self.cff.setRow(self.recording, self.segStart, 'boundary')
			if self.conditionId != None:
				if not self.conditionDict.has_key(self.conditionId):
					self.conditionDict[self.conditionId] = {'id': self.conditionId}
				self.cff.setRow(self.recording, self.segStart, 'condition', self.conditionDict[self.conditionId])
			if self.speakerId != None:
				if not self.speakerDict.has_key(self.speakerId):
					self.speakerDict[self.speakerId] = {'id': self.speakerId}
				self.cff.setRow(self.recording, self.segStart, 'speaker', self.speakerDict[self.speakerId])
			self.cff.setRow(self.recording, self.segStart, 'orth', self.orth)
			self.orth = None
		elif name == "segment":
			if self.segEnd != None:
				self.cff.setRow(self.recording, self.segEnd, 'boundary')
		elif name == 'corpus' or name == 'subcorpus':
			self.corpusNameList = self.corpusNameList[:-1]
		elif name == 'speaker-description':
			self.isDescription = False
			if not self.descriptionDict.has_key('id'):
				self.descriptionDict['id'] = self.descriptionId
			self.speakerDict[self.descriptionId] = self.descriptionDict
		elif name == 'condition-description':
			self.isDescription = False
			if not self.descriptionDict.has_key('id'):
				self.descriptionDict['id'] = self.descriptionId
			self.conditionDict[self.descriptionId] = self.descriptionDict

		if self.isDescription:
			if name != self.descriptionKey:
				raise "Error: tag depth > 1 in description"
			self.descriptionDict[name] = self.data


		self.data = ""

	def characters(self, chars):
		self.data += chars

###########################################################
# traceback parser

class TracebackHandler(ContentHandler):
	def __init__(self, cff = Cff()):
		self.cff = cff
		self.recording = None
		self.orth = None
		self.data = ""
		self.item = {}
		self.itemAttr = None
		self.itemList = []
		self.segStart = 0
		self.segEnd = 0
		self.segTrack = 0
		self.features = 0 # to be added if acoustic scores normalized with features should be used
	def startElement(self, name, attrs):
		dictAttrs = dict(attrs)
		if name == "condition":
			self.cff.setRow(self.recording, self.segStart, 'condition', dictAttrs)

		elif name == "features":
			self.item['features-start'] = int(dictAttrs["start"])
			self.item['features-end'] = int(dictAttrs["end"])

		elif name == "item":
			self.item = {}

		elif name == "recording":
			self.recording = dictAttrs["audio"]

		elif name == "samples":
			self.item['samples-start'] = dictAttrs["start"]
			self.item['samples-end'] = dictAttrs["end"]

		elif name == "segment":
			self.itemList = []
			self.segStart = dictAttrs["start"]
			self.segEnd = dictAttrs["end"]
			if self.segTrack != attrs.get('track', 0):
			   self.segTrack = attrs.get('track', 0)
			   self.cff.setRow(self.recording, self.segStart, 'track', self.segTrack)
			self.cff.setRow(self.recording, self.segStart, 'boundary')

		elif name == "score":
			if not self.item.has_key('score'):
				self.item['score'] = {}
			self.itemAttr = dictAttrs["type"]

		elif name == "speaker":
			self.cff.setRow(self.recording, self.segStart, 'speaker', dictAttrs)

	def endElement(self, name):
		self.data = self.data.replace("\n", " ")
		self.data = re.sub(" +", " ", self.data)
		self.data = re.sub("^ ", "", self.data)
		self.data = re.sub(" $", "", self.data)

		if name == "item":
			self.itemList.append(self.item)

		elif name == "orth":
			self.item['orth'] = self.data

		elif name == "phon":
			self.item['phon'] = self.data

		elif name == "segment":
			end = self.segEnd
			segFeatures = None
			segItemEndTime = None
			for item in self.itemList:
				if item.has_key('features-end'):
					if segFeatures == None or segFeatures < item['features-end']:
						segFeatures = item['features-end']
				if item.has_key('samples-end'):
					if segItemEndTime == None or segItemEndTime < item['samples-end']:
						segItemEndTime = eval(item['samples-end'])
			if segFeatures != None:
				featureDuration = (float(self.segEnd) - float(self.segStart))/(segFeatures + 1)
			if segItemEndTime != None and abs(float(self.segEnd) - segItemEndTime) > 1.0:
				print >> sys.stderr, "Warning: itemEnd and segEnd differ"

			for item in self.itemList:
				if item.has_key('samples-start'):
					itemStartTime = eval(item['samples-start'])
				elif item.has_key('features-start'):
					itemStartTime = float(self.segStart) + (item['features-start'] * featureDuration)
				else:
					# dirty hack to allow for Sentence-Boundaries which have no time information
					itemStartTime += 0.00001
					#itemStartTime = None
				if item.has_key('samples-end'):
					itemEndTime = eval(item['samples-end'])
				elif item.has_key('features-end'):
					itemEndTime = float(self.segStart) + ((item['features-end'] + 1) * featureDuration)
				else:
					itemEndTime = None
				if itemEndTime != None and itemStartTime != None:
					itemLength = itemEndTime - itemStartTime

				for key in item.keys():
					if key == 'score':
						self.cff.setRow(self.recording, itemStartTime, 'score', item['score'])
					if key == 'orth':
						self.appendOrth(item['orth'])
						if itemLength > 0:
							self.cff.setRow(self.recording, itemStartTime, 'orth', self.orth)
							self.orth = ''

			self.cff.setRow(self.recording, end, 'boundary')

		elif name == "score":
			self.item['score'][self.itemAttr] = str(round(float(self.data), 10))

		self.data = ""

	def appendOrth(self, orth):
		if self.orth == None:
			self.orth = orth+' '
		else:
			self.orth += orth+' '

	def characters(self, chars):
		self.data += chars

def tracebackParser(inputFile, cff = Cff()):
 parser	 = make_parser()

 handler = TracebackHandler(cff)
 parser.setContentHandler( handler )
 parser.parse( inputFile )
 return handler.cff

###########################################################
# trs parser

class TrsHandler(ContentHandler):
	def __init__(self, path, obj=Cff() ):
		self.cff = obj
		self.readPath = path
		self.recording = ''
		self.condition = None
		self.orth = None
		self.speaker = None
		self.speakerDict = {}
		self.data = ''
		self.segStart = 0
		self.segEnd = 0
		self.segTrack = 0
		self.syncTime = 0.0
		self.lastSegEndTime = 0.0
		self.isTrans = False
		self.isEpisode = False
		self.isSection = False
		self.isTurn = False
		self.isEvent = False

	def startElement(self, name, attrs):
		if name == 'Speaker':
			self.speakerDict[attrs.get('id')] = dict(attrs)

		if name == 'Topic' and 'desc' in attrs.getNames() and attrs.getValue('desc').find('/') > 1:
			self.data = ''

		elif name == 'Trans':
			if 'audio_filename' in attrs.getNames():
				self.recording = attrs.getValue('audio_filename')
			self.isTrans = True

		elif name == 'Episode':
			self.isEpisode = True

		elif name == 'Section':
			self.isSection = True

		elif name == 'Turn':
			self.isTurn = True
			if 'speaker' in attrs.getNames():
				self.speaker = attrs.getValue('speaker')
			else:
				self.speaker = None
			self.lastSegEndTime = attrs.getValue('endTime')
			if self.condition != None:
				self.cff.setRow(self.recording, attrs.getValue('startTime'), 'condition', {})
			if self.speaker != None:
				if self.speakerDict.has_key(self.speaker):
					self.cff.setRow(self.recording, attrs.getValue('startTime'), 'speaker', self.speakerDict[self.speaker])
				else:
					print >> sys.stderr, self.recording, attrs.getValue('startTime'), "speaker:", self.speaker.encode('UTF-8'), "unknown"
			else:
				self.cff.setRow(self.recording, attrs.getValue('startTime'), 'speaker', {})
			self.cff.setRow(self.recording, attrs.getValue('startTime'), 'boundary')

		elif name == 'Sync':
			self.orthUpdate()
			self.syncTime = attrs.getValue('time')
			self.cff.setRow(self.recording, self.syncTime, 'boundary')

		else:
			if self.isEpisode:
				self.data += ' <' + self.formTag(name, attrs.items()) + '/> '

	def endElement(self, name):
		if name == 'Trans':
			self.isTrans = False

		elif name == 'Episode':
			self.isEpisode = False

		elif name == 'Section':
			pass

		elif name == 'Turn':
			self.orthUpdate()
			self.cff.setRow(self.recording, self.lastSegEndTime, 'boundary')
			self.isTurn = False

	def characters(self, chars):
		if self.isTurn == True:
			self.data += chars

	def formTag(self, element, attr=[]):
		return string.join([element] + map(lambda kv: '%s="%s"' % kv, attr))

	def orthUpdate(self):
		self.data = self.data.replace("\n", " ")
		self.data = re.sub(" +", " ", self.data)
		self.data = re.sub("^ ", "", self.data)
		self.data = re.sub(" $", "", self.data)
		if self.data != "":
			self.cff.setRow(self.recording, self.syncTime, 'orth', self.data)
			self.data = ""

def trsParser(inFileName, cff = Cff()):
	parser = make_parser()
	handler = TrsHandler( os.path.dirname(inFileName), cff )
	parser.setFeature(feature_namespaces, False)
	parser.setFeature(feature_external_ges, False)
	parser.setFeature(feature_external_pes, False)
	parser.setContentHandler( handler )
	parser.parse( inFileName )
	return handler.cff

###########################################################
# bliss corpus writer

class CorpusPreProc:

	def __init__(self):
		self.reset()
		self.condition = {}
		self.speaker = {}

		#ugly hack: TODO don't discard previous segment end time, if a single boundary occurs (NOT two boundaries)
		#and the recording doesn't change: then, this time is the new start time for the next segment...
		#before this hack: there where segments with orthographies having the same start and end time....
		#self.previousEnd = None
		self.recording = None
		self.previousBoundaryTime = None
		self.boundaryTime = None

	def reset(self):
		self.start = None
		self.end = None
		self.orth = None

	def setKey(self, recording, time, type):
		if self.recording == None or self.recording != recording:
			self.recording = recording
			self.previousBoundaryTime = None
			self.boundaryTime = None

		if type == 'boundary':
			if self.boundaryTime != None:
				self.previousBoundaryTime = self.boundaryTime
				if self.previousBoundaryTime > time:
					raise('Error: start has to grow '+recording+' '+str(time)+' '+str(self.previousBoundaryTime))

			self.boundaryTime = time

			self.start = self.previousBoundaryTime
			self.end = self.boundaryTime

			#if self.end != None:
			#	self.start = self.end
			#	self.end = None
			#
			#if self.start == None:
			#	self.start = time
			#	if self.previousEnd != None:
			#		self.start = self.previousEnd
			#elif float(self.start) > time:
			#	raise('Error: start has to grow')
		#
		#	if self.end == None:
		#		self.end = time
		#		self.previousEnd = time
		#	elif float(self.end) <= time:
		#		self.end = time
		#		self.previousEnd = time
		#	else:
		#		raise('Error: end has to grow')

	def appendOrth(self, orth):
		if self.orth == None:
			self.orth = orth+' '
		else:
			self.orth += orth+' '

	def setSpeaker(self, speaker):
		self.speaker = speaker

	def extractId(self, dict):
		if dict.has_key('id'):
		    return(dict['id'])
		elif dict.has_key('name'):
		    return(dict['name'])
		else:
		    id = ''
		    for key in dict.keys():
			id += str(key)+str(dict[key])
		    return(id)

	def setCondition(self, condition):
		self.condition = condition

	#outlist=[recording, ]
	def getLine(self):
		#if None in [ self.recording, self.start, self.end, self.orth ]:
		#	raise('Error: attribute missing')
		outList = [ self.recording ]
		if self.speaker == None:
			outList.append('unknown')
		else:
			outList.append(self.speaker)
		if self.condition == None:
			outList.append('unknown')
		else:
			outList.append(self.condition)
		outList += [ str(self.start),str(self.end), self.orth[:-1] ]
		return outList

def blissCorpusWriter(cff, fileName = 'stdout'):
	currentRecording = None
	currentCorpusNameList = [ 'EPPS' ]

	xml = openXml(fileName, encoding='UTF-8')
	xml.openComment()
	xml.cdata(xml.escape('generated by ' + sys.argv[0] + ' on ' + str(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S"))))
	xml.closeComment()

	corpusPreProc = CorpusPreProc()

	# create speakerIdDict and conditionIdDict
	speakerIdDict = {}
	conditionIdDict = {}
	for row in cff.getRows():
		if row[2] == 'speaker':
			speakerId = corpusPreProc.extractId(row[3])
			speakerIdDict[speakerId] = row[3]

		elif row[2] == 'condition':
			conditionId = corpusPreProc.extractId(row[3])
			conditionIdDict[conditionId] = row[3]

		elif row[2] == 'corpus-name':
			currentCorpusNameList = row[3][:1]


	xml.open('corpus', { 'name': currentCorpusNameList[0]})
	# write speakerIdDict and conditionIdDict
	for speakerId in speakerIdDict.keys():
		xml.open('speaker-description', {'name': speakerId })
		for item in speakerIdDict[speakerId].keys():
			xml.open(item)
			xml.cdata(xml.escape(speakerIdDict[speakerId][item]))
			xml.close(item)
		xml.close('speaker-description')

	for conditionId in conditionIdDict.keys():
		xml.open('condition-description', {'name': conditionId })
		for item in conditionIdDict[conditionId].keys():
			xml.open(item)
			xml.cdata(xml.escape(conditionIdDict[conditionId][item]))
			xml.close(item)
		xml.close('condition-description')

	for row in cff.getRows():
		corpusPreProc.setKey(row[0], row[1], row[2])

		if row[2] == 'corpus-name':
			for name in currentCorpusNameList[1:]:
				xml.close('subcorpus')
			for name in row[3][1:]:
				xml.open('subcorpus', { 'name': name })
			currentCorpusNameList = row[3]

		if row[2] == 'boundary' and corpusPreProc.orth != None:
		   if not None in corpusPreProc.getLine():
			if currentRecording == None:
				currentRecording = corpusPreProc.getLine()[0]
				if currentRecording.startswith('/'):
					currentName = currentRecording[1:]
				else:
					currentName = currentRecording
					#print currentRecording
				xml.open('recording', {'audio': currentRecording, 'name':currentName})
			elif currentRecording != corpusPreProc.getLine()[0]:
				xml.close('recording')
				currentRecording = corpusPreProc.getLine()[0]
				if currentRecording.startswith('/'):
					currentName = currentRecording[1:]
				else:
					currentName = currentRecording
				xml.open('recording', {'audio': currentRecording, 'name':currentName})
			elif currentRecording == corpusPreProc.getLine()[0]:
				pass
			xml.open('segment', {'start': corpusPreProc.getLine()[3], 'end': corpusPreProc.getLine()[4], 'name': corpusPreProc.getLine()[3]+'-'+corpusPreProc.getLine()[4]})
			if corpusPreProc.getLine()[1] != {}:
				xml.empty('speaker', { 'name': corpusPreProc.extractId(corpusPreProc.getLine()[1]) })
			if corpusPreProc.getLine()[2] != {}:
				xml.empty('condition', {'name': corpusPreProc.extractId(corpusPreProc.getLine()[2]) })
			xml.open('orth')
			xml.cdata(corpusPreProc.getLine()[5])
			xml.close('orth')
			xml.close('segment')
			corpusPreProc.reset()
		   else:
			   print >> sys.stderr, corpusPreProc.getLine()

		elif row[2] == 'orth':
			corpusPreProc.appendOrth(row[3])

		elif row[2] == 'speaker':
					corpusPreProc.setSpeaker(row[3])

		elif row[2] == 'condition':
					corpusPreProc.setCondition(row[3])

	xml.close('recording')
	for name in currentCorpusNameList[1:]:
		xml.close('subcorpus')
	xml.close('corpus')
	closeXml(xml)

###########################################################
# trs writer

class TrsPreProc:

	def __init__(self):
		self.reset()
		self.speaker = None

	def reset(self):
		self.recording = None
		self.start = None
		self.end = None
		self.orth = None

	def setKey(self, recording, time):
		if self.recording == None:
			self.recording = recording
		elif self.recording != recording:
			raise('Error: recording can only be set once')
		if self.start == None:
			self.start = time
		elif float(self.start) > time:
			raise('Error: start has to grow')
		if self.end == None:
			self.end = time
		elif float(self.end) <= time:
			self.end = time
		else:
			raise('Error: end has to grow')

	def appendOrth(self, orth):
		if self.orth == None:
			self.orth = orth+' '
		else:
			self.orth += orth+' '

	def setSpeaker(self, speaker):
		self.speaker = speaker

	def getLine(self):
		if None in [ self.recording, self.start, self.end, self.orth ]:
			raise('Error: attribute missing')
		outList = [ self.recording ]
		if self.speaker == None:
			outList.append('unknown')
		else:
			outList.append(self.speaker)
		outList += [ str(self.start),str(self.end), self.orth[:-1] ]
		return outList

	def getTime(self):
		return self.start



def trsWriter(cff, file = sys.stdout):
	currentRecording = None
	speakerDict = {}
	speakerInfoDict = {}
	turnDict = {}
	firstStartFile = True
	firstStartTime = 0.0
	isSingleSpeaker = False
	encoder, decoder, streamReader, streamWriter = codecs.lookup('UTF-8')
	xml = XmlWriter(streamWriter(file), 'UTF-8')
	xml.setIndent_str('')
	xml.setMargin(1000000)
	xml.begin()
	xml.openComment()
	xml.cdata('generated by ' + sys.argv[0] + ' on ' + str(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))
	xml.closeComment()
	xml.cdata('<!DOCTYPE Trans SYSTEM "trans-13.dtd">')

	trsPreProc = TrsPreProc()
	# first pass to determine speakers and turns (and the recording name)
	for row in cff.getRows():
		if firstStartFile == True:
			firstStartTime = row[1]
			firstStartFile = False
		if row[2] == 'speaker':
			currentRecording = row[0]
			speakerDict = row[3]
			if speakerDict == {}:
				speakerInfoDict['unknown'] = {'id': 'unknown', 'name': 'unknown', 'check': '', 'type': '', 'dialect': '', 'accent': '', 'scope': ''}
			else:
				speaker = speakerDict['name']
				speakerInfoDict[speaker] = {'id': speaker, 'name': speaker, 'check': '', 'type': '', 'dialect': '', 'accent': '', 'scope': ''}
			if turnDict == {}:
				currentSpeaker = speaker
				speakerStart = row[1]
				turnDict[str(round(row[1], 3))] = 0
			if currentSpeaker != speaker:
				currentSpeaker = speaker
				turnDict[str(round(speakerStart, 3))] = str(round(row[1], 3))
				speakerStart = row[1]
				turnDict[str(round(row[1], 3))] = 0
			elif currentSpeaker == speaker:
				turnDict[str(round(speakerStart, 3))] = str(round(row[1], 3))
				speakerStart = row[1]
				turnDict[str(round(row[1], 3))] = 0
		if row[2] == 'boundary':
			lastKey = row[1]
	# if no speaker info is provided in the input file
	if speakerDict == {}:
		speakerInfoDict['unknown'] = {'id': 'unknown', 'name': 'unknown', 'check': '', 'type': '', 'dialect': '', 'accent': '', 'scope': ''}
		turnDict[str(round(firstStartTime, 3))] = str(round(lastKey, 3))
		isSingleSpeaker = True
	else:
		turnDict[str(round(speakerStart, 3))] = str(round(lastKey, 3))

	xml.open('Trans', {'audio_filename': currentRecording, 'version_date': str(datetime.datetime.now().strftime("%Y%m%d")), 'scribe': 'i6 RWTH Aachen', 'version': '1'})
	xml.open('Speakers')
	keys = speakerInfoDict.keys()
	keys.sort()
	for key in keys:
		xml.empty('Speaker', speakerInfoDict[key])
	xml.close('Speakers')

	# second pass
	keys = turnDict.keys()
	#print turnDict
	keys.sort(lambda a,b: cmp(float(a),float(b)))
	roundedKeys = []
	for key in keys:
		currentKey = str(round(float(key),3))
		roundedKeys += [currentKey]
	xml.open('Episode')
	xml.open('Section', {'type': 'report', 'startTime': keys[0], 'endTime': lastKey})
	if isSingleSpeaker:
		xml.open('Turn', {'speaker': 'unknown', 'startTime': keys[0], 'endTime': turnDict[str(round(float(keys[0]), 3))]})
	for row in cff.getRows():
		trsPreProc.setKey(row[0], row[1])
		if row[2] == 'boundary' and trsPreProc.orth != None:
			xml.cdata(trsPreProc.getLine()[4])#.encode('UTF-8'))
			if trsPreProc.getLine()[3] not in roundedKeys:
				xml.empty('Sync', {'time': trsPreProc.getLine()[3],})
			else:
				pass
			trsPreProc.reset()
		elif row[2] == 'boundary':
			xml.empty('Sync', {'time': trsPreProc.getTime(),})

		if row[2] == 'orth':
			trsPreProc.appendOrth(row[3])

		if row[2] == 'speaker':
			speakerDict = row[3]
			if speakerDict == {}:
				trsPreProc.setSpeaker('unknown')
			else:
				#trsPreProc.setSpeaker(speakerDict['id'])
				trsPreProc.setSpeaker(speakerDict['name'])
			if str(round(float(row[1]), 3)) == str(round(float(keys[0]), 3)):
				if speakerDict == {}:
					xml.open('Turn', {'startTime': keys[0], 'endTime': turnDict[str(round(float(keys[0]), 3))]})
					xml.empty('Sync', {'time': keys[0]})
				else:
					xml.open('Turn', {'speaker':  speakerDict['name'], 'startTime': keys[0], 'endTime': turnDict[str(round(float(keys[0]), 3))]})
					xml.empty('Sync', {'time': keys[0]})
			else:
				if speakerDict == {}:
					#print 'close'
					#print 'open'
					#xml.close('Turn')
					#xml.open('Turn', {'startTime': row[1], 'endTime': turnDict[str(round(float(row[1]), 3))]})
					#xml.empty('Sync', {'time': row[1]})
					pass
				else:
					xml.close('Turn')
					xml.open('Turn', {'speaker':  speakerDict['name'], 'startTime': row[1], 'endTime': turnDict[str(round(float(row[1]), 3))]})
					#xml.empty('Sync', {'time': row[1]})
	xml.close('Turn')
	xml.close('Section')
	xml.close('Episode')
	xml.close('Trans')


###########################################################
# stm writer
class StmRow:
	def __init__(self):
		self.reset()
	def reset(self):
		self.recording = None
		self.track = None
		self.speaker = None
		self.start = None
		self.end = None
		self.condition = None
		self.orth = None
	def setKey(self, recording, time):
		if self.recording == None:
			self.recording = recording
		elif self.recording != recording and self.orth != None:
			print >> sys.stderr, 'Error: recording finished without a boundary but with orth (words), which will be ignored'
			self.reset()
		if self.start == None:
			self.start = time
		elif float(self.start) > time:
			raise('Error: start has to grow')
		if self.end == None:
			self.end = time
		elif float(self.end) <= time:
			self.end = time
		else:
			raise('Error: end has to grow')
	def appendOrth(self, orth):
		if orth != None:
			if self.orth == None:
				self.orth = orth+' '
			else:
				self.orth += orth+' '
	def getLine(self):
		if None in [ self.recording, self.start, self.end, self.orth ]:
			raise('Error: attribute missing')
		outList = [ self.recording ]
		if self.track == None:
			outList.append('0')
		else:
			outList.append(str(self.track))
		if self.speaker == None:
			outList.append('unknown')
		else:
			outList.append(self.speaker)
		outList += [ str(self.start), str(self.end) ]
		if self.condition == None:
			outList.append('unknown')
		else:
			outList.append(self.condition)

		outList.append(self.orth[:-1])
		return ' '.join(outList)

def stmWriter(cff, file = sys.stdout):
	stmRow = StmRow()
	for row in cff.getRows():
		stmRow.setKey(row[0], row[1])

		if row[2] == 'boundary':
			if stmRow.orth != None:
				print >> file, stmRow.getLine().encode('UTF-8')
			stmRow.reset()

		if row[2] == 'orth':
			stmRow.appendOrth(row[3])

		if row[2] == 'speaker':
			if row[3].has_key('name'):
				stmRow.speaker = row[3]['name']

		if row[2] == 'condition':
			if row[3].has_key('name'):
				stmRow.condition = row[3]['name']

		if row[2] == 'track':
			stmRow.track = row[3]


###########################################################
# ctm writer

def ctmWriter(cff, file = sys.stdout):
	ctmRow = CtmRow()
	for row in cff.getRows():
		ctmRow.setKey(row[0], row[1])

		if row[2] == 'boundary':
			if ctmRow.orth != None:
				print >> file, ctmRow.getLine().encode('UTF-8')
			ctmRow.reset()

		if row[2] == 'score' and row[3].has_key('confidence'):
			ctmRow.setConf(row[3]['confidence'])
		if row[2] == 'orth':
			ctmRow.appendOrth(row[3])


class CtmRow:
	def __init__(self):
		self.reset()
	def reset(self):
		self.recording = None
		self.track = None
		self.start = None
		self.end = None
		self.orth = None
		self.conf = None
	def setKey(self, recording, time):
		if self.recording == None:
			self.recording = recording
		elif self.recording != recording:
			raise('Error: recording can only be set once: '+str(recording)+' '+str(time))
		if self.start == None:
			self.start = time
		elif float(self.start) > time:
			raise('Error: start has to grow')
		if self.end == None:
			self.end = time
		elif float(self.end) <= time:
			self.end = time
		else:
			raise('Error: end has to grow')
	def appendOrth(self, orth):
		if self.orth == None:
			self.orth = orth+' '
		else:
			self.orth += orth+' '
	def setConf(self, conf):
		self.conf = conf
	def getLine(self):
		if None in [ self.recording, self.start, self.end, self.orth ]:
			raise('Error: attribute missing')
		outList = [ self.recording ]
		if self.track == None:
			outList.append('1')
		else:
			outList.append(self.track)
		outList += [ str(self.start), str(round(float(self.end)-float(self.start),4)), self.orth ]
		if self.conf != None:
			outList.append(str(self.conf))
		return ' '.join(outList)

################################################################
# stm parser

def stmParser(inFileName, cff = Cff()):
	file = open(inFileName, 'r')
	for line in file.readlines():
		line = line[:-1].decode('UTF-8')
		if not line.startswith(';;'):
			fields = line.split(' ')
			recording = fields[0]
			channel = fields[1]
			speaker = fields[2]
			segStart = fields[3]
			segEnd = fields[4]
			condition = fields[5]
			orth = ' '.join(fields[6:])

			if speaker not in [ 'excluded_region', 'inter_segment_gap']:
				cff.setRow(recording, segStart, 'boundary')
				cff.setRow(recording, segStart, 'speaker', { 'name': speaker } )
				cff.setRow(recording, segStart, 'condition', { 'name': condition } )
				cff.setRow(recording, segStart, 'track', int(channel) )
				cff.setRow(recording, segStart, 'orth', orth )
				cff.setRow(recording, segEnd, 'boundary')
	return cff
	#cff.setRow(self.recording, self.segStart, 'condition', dictAttrs)

################################################################
# uem parser

def uemParser(inFileName, cff = Cff()):
	file = open(inFileName, 'r')
	preLenFields = None
	for line in file.readlines():
		line = line[:-1].decode('UTF-8')
		if not line.startswith(';;'):
			fields = line.split(' ')
			recording = fields[0]
			channel = None
			if len(fields) == 4:
				channel = fields[1]
				segStart = fields[2]
				segEnd = fields[3]
			elif len(fields) == 3:
				segStart = fields[1]
				segEnd = fields[2]
			else:
				raise('Error: expected either 3 or 4 fields per line: '+str(line))
			if preLenFields != None and preLenFields != len(fields):
				raise('Error: expected either 3 or 4 fields per line: '+str(line))
			preLenFields = len(fields)
			orth = ' '

			cff.setRow(recording, segStart, 'boundary')
			if channel != None:
				cff.setRow(recording, segStart, 'track', int(channel) )
			cff.setRow(recording, segStart, 'orth', orth )
			cff.setRow(recording, segEnd, 'boundary')
	return cff


################################################################
# ctm parser

def ctmParser(inFile, cff=Cff()):
    file = open(inFile, 'r')
    line = file.readline()
    # delete newline and decode
    line = line[:-1].decode('UTF-8')
    # split columns into fields
    fields = line.split(' ')
    # if last col is Float number => set flag
    try:
	float(fields[-1])
	useConfidence = True
    except ValueError:
	useConfidence = False
    for newline in file.readlines():
	# delete newline and decode
	newline = newline[:-1].decode('UTF-8')
	newfields = newline.split(' ')
	# get fields
	recording = fields[0]
	segStart = fields[2]
	segEnd = str(float(fields[2])+float(fields[3]))
	# delete last column if float
	if useConfidence:
	    confidence = fields[-1]
	    del fields[-1]
	orth = ' '.join(fields[4:])
	# write into cff
	cff.setRow(recording, segStart, 'boundary')
	cff.setRow(recording, segStart, 'orth', orth)
	if useConfidence: cff.setRow(recording, segStart, "score {u'confidence': '"+confidence+"'}")
	cff.setRow(recording, segEnd, 'boundary')
	# set newline/fields as old line/fields
	line = newline
	fields = newfields
    recording = fields[0]
    segStart = fields[2]
    # get number of positions after decimal point
    length = len(segStart) - len(str(int(float(segStart)))) - 1
    # if there are <=3 positions after dec. point ==> round new values
    if length <= 3:
	segStart = "%(end).3f" % {'end': eval(segStart)}
	segEnd = "%(end).3f" % {'end': (eval(segStart)+eval(fields[3]))}
    else:
	segEnd = str(eval(segStart)+eval(fields[3]))
    # delete last column if float
    if useConfidence:
	confidence = fields[-1]
	del fields[-1]
    orth = ' '.join(fields[4:])
    # write into cff
    cff.setRow(recording, segStart, 'boundary')
    cff.setRow(recording, segStart, 'orth', orth)
    if useConfidence: cff.setRow(recording, segStart, 'score' , "{u'confidence': '"+confidence+"'}")
    cff.setRow(recording, segEnd, 'boundary')
    return cff

################################################################
# statistics

class Statistics:
	def __init__(self):
		self.histogram = {}
		self.bigSegmentsDict = {}
		self.recording = None
		self.recordingCounter = 0
		self.segmentCounter = 0
		self.segmentDurationSum = 0.0
		self.segmentDuration = 0.0
		self.bigSegmentsDurationSum = 0.0
		self.start = None
		self.end = None
		self.reset()

	def reset(self):
		if self.start != None and self.end != None:
			self.duration = float(self.end) - float(self.start)
			if options.bigSegments:
				if self.duration >= options.bigSegments:
					self.bigSegmentsDict[(self.recording, self.start, self.end, self.duration)] = '0'
					self.bigSegmentsDurationSum += self.duration
			self.segmentCounter += 1
			self.segmentDurationSum += self.duration
			if options.binSize:
				self.histogram_insert(self.duration, self.histogram, options.binSize)
		self.start = None
		self.end = None
		self.orth = None

	def setValues(self, recording, time):
		if self.recording != recording:
			self.recording = recording
			self.recordingCounter+=1
			self.start = None
			self.end = None
		if self.start == None:
			self.start = time
		elif float(self.start) > time:
			raise('Error: start has to grow '+recording+' '+str(time))
		if self.end == None:
			self.end = time
		elif float(self.end) <= time:
			self.end = time
		else:
			raise('Error: end has to grow '+recording+' '+str(time))

	def appendOrth(self, orth):
		if self.orth == None:
			self.orth = orth+' '
		else:
			self.orth += orth+' '

	def getLine(self):
		if None in [ self.start, self.end, self.orth ]:
			raise('Error: attribute missing')
		outList = [ self.recording ]
		if self.track == None:
			outList.append('0')
		else:
			outList.append(self.track)
		if self.speaker == None:
			outList.append('unknown')
		else:
			outList.append(self.speaker)
		outList += [ str(self.start), str(self.end), self.orth[:-1] ]
		return ' '.join(outList)

	def histogram_insert(self, duration, hist_Dict, binSize):
		bucket = 0.0
		while (float(duration) > float(bucket)):
			bucket += binSize
		if hist_Dict.has_key(bucket-binSize):
			hist_Dict[bucket-binSize] += 1
		else:
			hist_Dict[bucket-binSize] = 1

	def histogram_output(self, hist_Dict, binSize):
		buckets = hist_Dict.keys()
		buckets.sort()
		key= 0.0
		#if empty buckets should be reported, uncomment this region
		#while key < buckets[-1]:
		#	if hist_Dict.has_key(key):
		#		pass
		#	else:
		#		hist_Dict[key]=0
		#	key+= binSize
		#buckets = hist_Dict.keys()
		#buckets.sort()
		for item in buckets:
			print str(item) +' '+str(hist_Dict[item])

	def getBigSegments(self, seg_dict):
		keys = seg_dict.keys()
		keys.sort()
		for item in keys:
			print str(item[0]) + '	' + str(round(float(item[1]),3)) + '\t' + str(round(float(item[2]),3)) + '	   \t' + str(round(float(item[3]),3))

	def getTimeInHours(self, seconds):
		hours = int(seconds / 3600)
		minutes = int((seconds - 3600 * hours)/ 60 )
		seconds = round(float(seconds - hours * 3600 - minutes * 60),2)
		return(str(str(hours) + 'h ' + str(minutes) + 'm ' + str(seconds) + 's'))

	def getPercent(self, w, g):
		return str(round(float(float(w)/float(g) * 100),2))

def getStatistics(cff, options):
	statistics = Statistics()
	for row in cff.getRows():
		statistics.setValues(row[0], row[1])
		if row[2] == 'boundary' and statistics.orth != None:
			statistics.reset()
		if row[2] == 'orth':
			statistics.appendOrth(row[3])
	print '***********************************************************************'
	print '* Statistics:\n*'
	print '* # Recordings: ' + str(statistics.recordingCounter)
	print '* # Segments: ' + str(statistics.segmentCounter)
	print '* sum of all Segment durations: ' + str(statistics.segmentDurationSum) + 's ('+ statistics.getTimeInHours(statistics.segmentDurationSum) + ')'
	if options.bigSegments:
		print '* sum of all Segment durations >= ' + str(options.bigSegments) + 's: ' + str(statistics.bigSegmentsDurationSum) + 's (' + statistics.getTimeInHours(statistics.bigSegmentsDurationSum) + ', ' + statistics.getPercent(statistics.bigSegmentsDurationSum, statistics.segmentDurationSum) + '%)'
	print '* average Segment length: ' + str(round(float(statistics.segmentDurationSum)/float(statistics.segmentCounter),2)) + 's'
	print '***********************************************************************'
	if options.binSize:
		print 'Histogram over the segment lengths with binsize '+ str(options.binSize) +'s :'
		statistics.histogram_output(statistics.histogram, options.binSize)
		print '*****************************************'
	if options.bigSegments:
		if statistics.bigSegmentsDict != {}:
			print 'Segments >= '+ str(options.bigSegments) + 's	 (recording, segment start, segment end, segment duration):\n'
			statistics.getBigSegments(statistics.bigSegmentsDict)
			print '****************************************************************************'
		else:
			print 'Segments >= '+ str(options.bigSegments) + 's (recording, segment start, segment end, segment duration):\n'
			print 'No segments are >= ' + str(options.bigSegments) +'s'
			print '****************************************************************************'



################################################################
optParser = OptionParser(usage="usage: %prog [OPTION...] [FILE...]", version="%prog 0.1")
optParser.set_description("Corpus conversion tool")
optParser.add_option("-o", "--output-file", metavar="FILE", action="store", type="string",
					 dest="outputFile", help="Set the output file", default='stdout')
optGroupE = OptionGroup( optParser, 'Corpus format options' )
optGroupE.set_description("Input formats: cff, ctm, stm, uem, bliss, trs, traceback \t\t\t Output formats: cff, ctm, stm, bliss, trs")
optGroupE.add_option("-f", "--input-format", metavar="FORMAT", action="store", type="string",
					 dest="inputFormat", help="Set the input file format, default: cff", default='cff')
optGroupE.add_option("-t", "--output-format", metavar="FORMAT", action="store", type="string",
					 dest="outputFormat", help="Set the output file format, default: cff", default='cff')
optGroupS = OptionGroup( optParser, 'Statistics options' )
optGroupS.add_option("-s", "--statistics", action="store_true",
					 dest="isStatistics", help="enable statistics")
optGroupS.add_option("-b", "--binsize", metavar="BINSIZE", action="store", type="float",
					 dest="binSize", help="histogram over the segment lengths with a given binsize")
optGroupS.add_option("-l", "--long-segments", metavar="MINLENGTH", action="store", type="float",
					 dest="bigSegments", help="segment facts about segments longer than MINLENGTH")
optParser.add_option_group(optGroupE)
optParser.add_option_group(optGroupS)
options, args = optParser.parse_args()


##############
#corpus input
##############

cff = Cff()
if len(args) == 0:
	if options.inputFormat == 'cff':
		cff.readRows(sys.stdin)
	else:
	   print >> sys.stderr, 'Input format:', options.inputFormat, 'not supported for stdin'
else:
	for arg in args:
		if options.inputFormat == 'bliss':
					cff = blissCorpusParser(arg, cff)
		elif options.inputFormat == 'traceback':
					cff = tracebackParser(arg, cff)
		elif options.inputFormat == 'cff':
			cff.readRows(open(arg, 'rb'))
		elif options.inputFormat == 'trs':
			cff = trsParser(arg, cff)
		elif options.inputFormat == 'stm':
			cff = stmParser(arg, cff)
		elif options.inputFormat == 'ctm':
			cff = ctmParser(arg, cff)
		elif options.inputFormat == 'uem':
			cff = uemParser(arg, cff)
		else:
			print >> sys.stderr, 'Unknown input format:', options.inputFormat

##############
#corpus statistics
##############

if options.isStatistics:
	getStatistics(cff, options)

###############
#corpus output
###############

if options.outputFormat == 'cff':
	if options.outputFile == 'stdout':
		file = sys.stdout
	else:
		file = open(options.outputFile, 'wb')
	cff.writeRows(file)
elif options.outputFormat == 'ctm':
	if options.outputFile == 'stdout':
		file = sys.stdout
	else:
		file = open(options.outputFile, 'wb')
	ctmWriter(cff, file)
elif options.outputFormat == 'stm':
	if options.outputFile == 'stdout':
		file = sys.stdout
	else:
		file = open(options.outputFile, 'wb')
	stmWriter(cff, file)
elif options.outputFormat == 'bliss':

	blissCorpusWriter(cff, options.outputFile)
elif options.outputFormat == 'trs':
	if options.outputFile == 'stdout':
		file = sys.stdout
	else:
		file = open(options.outputFile, 'wb')
	trsWriter(cff, file)
else:
	print >> sys.stderr, 'Unknown output format:', options.outputFormat
