#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

from optparse import OptionParser, OptionGroup
import sys
import string
import re
import gzip
import os.path
import codecs
import cPickle
from xml.sax import make_parser
from xml.sax.handler import ContentHandler

#########################################################################
# corpus format file (cff) object

class Cff:
	def __init__(self):
		self.cffDict = {}

	def setRow(self, recording, timePos, key, value = None):
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


#########################################################################
# segmentation strategies

class SimpleSeg:

	def __init__(self):
		self.isNewRecording = True
		self.reset()

	def reset(self):
		self.recording = None
		self.track = None
		self.speaker = None
		self.start = None
		self.end = None
		self.condition = None
		self.orth = None
		self.conf = None
		self.confCounter = None

	def setKey(self, recording, time):
		if self.recording == None:
			self.recording = recording
			self.isNewRecording = True
		elif self.recording != recording:
			self.recording = recording
			self.isNewRecording = True
		elif self.recording == recording:
			self.isNewRecording = False
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

	def setConf(self, conf):
		self.conf = float(conf)

	def addConf(self, conf):
		if self.conf == None:
			self.conf = float(conf)
			self.confCounter = 1
		else:
			self.conf += float(conf)
			self.confCounter += 1

	def getNormalizedConf(self):
		if self.conf != None:
			return (float(self.conf)/float(self.confCounter))
			#return float((float(self.conf)/float(self.confCounter)) * (float(self.conf)/float(self.confCounter)))
		else:
			return None

	def appendOrth(self, orth):
		if self.orth == None:
			self.orth = orth+' '
		else:
			self.orth += orth+' '

	def getOrth(self):
		return self.orth

	def getLine(self):
		if None in [ self.recording, self.start, self.end, self.orth ]:
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

def insertSync(cff):
	syncCff = Cff()
	boundaryRecording = None
	boundaryTime = None

	for row in cff.getRows():

		if row[2] == 'boundary':
			boundaryRecording = row[0]
			boundaryTime = row[1]
			syncCff.setRow(row[0], row[1], row[2])

		else:
			if row[0] == boundaryRecording and row[1] == boundaryTime:
				syncCff.setRow(row[0], row[1], row[2], row[3])
			else:
				syncCff.setRow(row[0], row[1], row[2], row[3])
				syncCff.setRow(row[0], row[1], 'boundary_sync')
	return syncCff

def insertSync_old(cff):
	syncCff = Cff()
	recording = None
	recordingTime = None
	newrecording = True
	rows = cff.getRows()
	rowCounter = -1

	for row in cff.getRows():
		rowCounter +=1
		if recording == None:
			newRecording = True
			recording = row[0]
			recordingTime = row[1]
		elif row[0] != recording:
			newRecording == True
			recording = row[0]
			recordingTime = row[1]
		elif row[0] == recording and row[1] != recordingTime:
			newRecording = False

		if row[2] == 'boundary' and newRecording == False:
			if (rowCounter+1)<len(rows):
				if rows[rowCounter+1][2] == 'boundary':
					syncCff.setRow(row[0], row[1], row[2], row[3])
					syncCff.setRow(rows[rowCounter+1][0], rows[rowCounter+1][1], rows[rowCounter+1][2], rows[rowCounter+1][3])
				else:
					syncCff.setRow(row[0], row[1], 'boundary')
			else:
				syncCff.setRow(row[0], row[1], row[2], row[3])
			newRecording = True
		elif row[1] != recordingTime and newRecording == False:
			syncCff.setRow(row[0], row[1], row[2], row[3])
			if not isBoundary:
				syncCff.setRow(row[0], row[1], 'boundary_sync')
		else:
			syncCff.setRow(row[0], row[1], row[2], row[3])
	return syncCff

def removeSync(cff):
	noSyncCff = Cff()
	for row in cff.getRows():
		if row[2] == 'boundary_sync':
			pass
		else:
			noSyncCff.setRow(row[0], row[1], row[2], row[3])
	return noSyncCff

def removeEmptySegs(cff):
	validCff = Cff()
	rows = cff.getRows()
	isBoundary = False
	maximum = len(rows)-1
	for row in range(len(rows)):
		if rows[row][2] == 'boundary':
			if isBoundary == True and row < maximum and rows[row+1][2] == 'boundary':
				pass
			elif isBoundary == True and row == maximum:
				pass
			elif row == 0 and rows[row+1][2] == 'boundary':
				isBoundary = True
			elif isBoundary == False:
				isBoundary = True
				validCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
			else:
				validCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
		else:
			validCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
			isBoundary = False
	return validCff

# minimal segmentation
def minimalSegmentation(cff):
	syncCff = insertSync(cff)
	minimalCff = Cff()
	for row in syncCff.getRows():
		if row[2] == 'boundary_sync':
			minimalCff.setRow(row[0], row[1], 'boundary')
		else:
			minimalCff.setRow(row[0], row[1], row[2], row[3])
	return minimalCff

# segmentation at silence tokens of min. length silenceLength
def rawSegmentation(cff, silenceLength=5.0):
	syncCff = insertSync(cff)
	pat = re.compile('(boundary|boundary_sync)')
	newCff = Cff()
	simpleSeg = SimpleSeg()
	cuttingPoints = {}
	cuttingEnds = {}
	segTokenList = ["[SILENCE]", "[silence]"] #, "[PAUSE]", "[pause]", "[ARTIC]", "[LAUGH]", "[THROAT]", "[B]", "[BREATH]", "[NOISE]", "[APPLAUSE]", "[MUSIC]", "[RUSTLE]", "[SOUND]", "[VOICE]"]
	segTokenDict = {}
	endCut = -1
	startCut = -1
	# first pass to determine cutting-points
	for row in syncCff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if re.match(pat, row[2]) and simpleSeg.orth != None:
			if simpleSeg.orth.strip() in segTokenList:
				length = (simpleSeg.end - simpleSeg.start)
				if length > silenceLength:
					cuttingPoints[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
					segTokenDict[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.orth.strip()
					cuttingEnds[simpleSeg.end] = None
			simpleSeg.reset()
		if row[2] == 'orth':
			simpleSeg.appendOrth(row[3])

	# second pass
	for row in syncCff.getRows():
		if re.match(pat, row[2]):
			if (row[0],row[1]) in cuttingPoints.keys():
				if str(row[2]) == 'boundary':
					newCff.setRow(row[0],row[1],row[2],row[3])
				startCut = row[1]
				endCut = cuttingPoints[(row[0],row[1])]
				cut = startCut + (endCut - startCut) / 2
				cut = str(round(cut, 4))
				newCff.setRow(row[0], row[1], 'orth', segTokenDict[(row[0],row[1])])
				newCff.setRow(row[0], cut , 'boundary')
				newCff.setRow(row[0], cut, 'orth', segTokenDict[(row[0],row[1])])
				newCff.setRow(row[0], cut, 'score', {u'confidence': '0.0', u'total': '0.0'})
			else:
				if row[1] in cuttingEnds.keys():
					if row[2] == 'boundary':
						newCff.setRow(row[0], row[1], row[2])
					else:
						pass
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
		#elif row[1] <= endCut:
		#	print row
		#	pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])

	newCff = removeSync(newCff)
	return newCff

#segmentation at cumulative non-speech token lengths > threshold
def advancedSegmentation(cff, silenceLength=5.0):
	pat = re.compile('(boundary|boundary_sync)')
	newCff = Cff()
	simpleSeg = SimpleSeg()
	cuttingPoints = {}
	cuttingEnds = {}
	segTokenList = ["[SILENCE]", "[silence]", "[PAUSE]", "[pause]", "[ARTIC]", "[LAUGH]", "[THROAT]", "[B]", "[BREATH]", "[NOISE]", "[APPLAUSE]", "[MUSIC]", "[RUSTLE]", "[SOUND]", "[VOICE]"]
	segTokenDict = {}
	endCut = -1
	startCut = -1
	tmpRow = -1

	rows = cff.getRows()

	marked = False
	# first pass to determine cutting-points
	for row in range(len(rows)):

		if row > tmpRow:
			marked = False

		if rows[row][2] == 'orth' and marked == False:
			if rows[row][3].strip() in segTokenList:

				simpleSeg.reset()
				simpleSeg.setKey(rows[row][0], rows[row][1])
				simpleSeg.appendOrth(rows[row][3])
				tmpRow = row
				isNonSpeech = True
				cutStart = None
				cutToken = None
				tmpTimeList=[(str(rows[row][1]), rows[row][3])]

				while ( (tmpRow <= len(rows)) and (isNonSpeech == True) and (rows[row][0] == simpleSeg.recording) and (rows[tmpRow][2] != 'boundary') ):
					tmpRow+=1
					if rows[tmpRow][2] == 'orth':
						simpleSeg.setKey(rows[tmpRow][0], rows[tmpRow][1])
						if (rows[tmpRow][3].strip() in segTokenList):
							simpleSeg.appendOrth(rows[tmpRow][2])
							tmpTimeList.append((str(rows[tmpRow][1]), rows[tmpRow][3]))
						else:
							simpleSeg.appendOrth(rows[tmpRow][2])
							tmpTimeList.append((str(rows[tmpRow][1]), rows[tmpRow][3]))
							isNonSpeech = False
					marked = True
				#print tmpTimeList
				length = (simpleSeg.end - simpleSeg.start)
				if length > silenceLength:
					cut = simpleSeg.start + (simpleSeg.end - simpleSeg.start) / 2
					if len(tmpTimeList) > 1:
						for i in range(len(tmpTimeList)-1):
							if (float(tmpTimeList[i][0]) <= float(cut)) and (float(tmpTimeList[i+1][0]) > float(cut)):
								cutStart = str(tmpTimeList[i][0])
								cutToken = tmpTimeList[i][1]
								break
							else:
								pass

						cuttingPoints[(simpleSeg.recording, str(cutStart))] = cut
						segTokenDict[(simpleSeg.recording, str(cutStart))] = cutToken
						cuttingEnds[simpleSeg.end] = None
						#print length," ",simpleSeg.start," ", simpleSeg.end," ",(simpleSeg.recording, cutStart)," ",cut," ",cutToken
					#else:
					#	cuttingPoints[(simpleSeg.recording, simpleSeg.start)] = cut
					#	segTokenDict[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.orth.strip()
					#	cuttingEnds[simpleSeg.end] = None
					#	print length," ",simpleSeg.start," ", simpleSeg.end," ",(simpleSeg.recording, cutStart)," ",cut," ",cutToken
			#print cuttingPoints



	# second pass
	for row in rows:
		if (row[0],str(row[1])) in cuttingPoints.keys():
			if str(row[2]) == 'boundary' or row[2] == 'score':
				newCff.setRow(row[0],row[1],row[2],row[3])
			cut = cuttingPoints[(row[0], str(row[1]))]
			newCff.setRow(row[0], row[1], 'orth', segTokenDict[(row[0],str(row[1]))])
			newCff.setRow(row[0], cut , 'boundary')
			newCff.setRow(row[0], cut, 'orth', segTokenDict[(row[0],str(row[1]))])
			newCff.setRow(row[0], cut, 'score', {u'confidence': '0.0', u'total': '0.0'})
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
#			if row[1] in cuttingEnds.keys():
#				if row[2] == 'boundary':
#					newCff.setRow(row[0], row[1], row[2])
#				else:
#			   		pass
#			else:
#				newCff.setRow(row[0],row[1],row[2],row[3])
#				endCut = -1.0
	return newCff

def advancedSegmentationWrapper(cff, silenceLength=5.0, segmentLength=40.0):

	class SegmentBoundaries:
		def __init__(self):
			self.recording = None
			self.start = None
			self.end = None
			self.reset()

		def reset(self):
			if self.start != None and self.end != None:
				self.duration = float(self.end) - float(self.start)
			self.start = None
			self.end = None
			self.orth = None

		def setValues(self, recording, time):
			if self.recording != recording:
				self.recording = recording
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

	def buildPartialCff(cff, segmentDict):
		currentCff = Cff()
		printRow = False
		rows = cff.getRows()
		keys = segmentDict.keys()
		keys.sort(lambda a,b: cmp( (str(a[0]),float(a[1])) ,(str(b[0]), float(b[1])) ))

		for row in range(len(rows)):
			if keys != []:
				if (keys[0][0], keys[0][1]) == (rows[row][0], rows[row][1]):
					currentCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
					printRow = True
				elif ( (keys[0][0], keys[0][2]) == (rows[row][0], rows[row][1]) ) and rows[row][2] == 'boundary':
					currentCff.setRow(rows[row][0], rows[row][1], rows[row][2])
					del keys[0]
					printRow = False
				elif printRow == True:
					currentCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
				else:
					pass
		return currentCff

	inputCff = cff
	tmpCff = Cff()
	partialCff = Cff()
	segmentBoundaries = SegmentBoundaries()
	segmentsTooBig = True
	trySmallerSilenceLength = True
	segmentDict = {}
	countDict = {}
	countBigSegs = 0

	tmpCff = advancedSegmentation(inputCff, silenceLength)
	partialCff = tmpCff

	while segmentsTooBig and trySmallerSilenceLength:

		segmentDict = {}
		countDict = {}
		countBigSegs = 0

		rows = partialCff.getRows()
		for row in rows:
			segmentBoundaries.setValues(row[0], row[1])
			if row[2] == 'boundary' and segmentBoundaries.orth != None:
				segmentDict[(segmentBoundaries.recording, segmentBoundaries.start, segmentBoundaries.end)] = None
				segmentBoundaries.reset()
			if row[2] == 'orth':
				segmentBoundaries.appendOrth(row[3])

		keys = segmentDict.keys()
		for key in keys:
			if float(key[2]-key[1]) > float(segmentLength):
				countBigSegs+=1
			else:
				del segmentDict[key]
		if countBigSegs > 0:
			segmentsTooBig = True
		else:
			segmentsTooBig = False
		partialCff = buildPartialCff(partialCff, segmentDict)

		silenceLength -= 0.1
		#print silenceLength
		if silenceLength < 0.1:
			trySmallerSilenceLength = False
		else:
			trySmallerSilenceLength = True

		print >> sys.stderr, "using nonSpeech length %f on %i segments" %(silenceLength, countBigSegs)
		partialCff = advancedSegmentation(partialCff, silenceLength)
		for row in partialCff.getRows():
			tmpCff.setRow(row[0], row[1], row[2], row[3])

	return tmpCff



#########################################################################
# filter strategies

# removal of segments with silence between two words
def removeSegsWithSil(cff, isStatistics=False):
	syncCff = cff #insertSync(cff)
	newCff = Cff()
	simpleSeg = SimpleSeg()
	pat = re.compile('[A-Za-z_\'] \[SILENCE\] [A-Za-z_\']')
	cuttingPoints = {}
	cuttingEnds = {}
	endCut = -1
	# first pass to determine segments to be discarded
	for row in syncCff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if row[2] == 'boundary' and simpleSeg.orth != None and row[1] > endCut:
			if re.search(pat, str(simpleSeg.orth.encode('UTF-8'))) != None:
				cuttingPoints[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
				cuttingEnds[simpleSeg.end] = None
			simpleSeg.reset()
		if row[2] == 'orth':
			simpleSeg.appendOrth(row[3])

	# second pass
	for row in syncCff.getRows():
		if row[2] == 'boundary':
			if (row[0],row[1]) in cuttingPoints.keys():
				endCut = cuttingPoints[(row[0],row[1])]
			else:
				if row[1] in cuttingEnds.keys():
					pass
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
		elif row[1] <= endCut:
			pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	if isStatistics:
		simpleFilterStats(cuttingPoints, syncCff)
	return newCff

# removal of segments with a confidence score lower than threshold
def removeSegsWithConf(cff, threshold=0.5, isStatistics=False):
	oldCff = cff
	newCff = Cff()
	simpleSeg = SimpleSeg ()
	cuttingPoints = {}
	cuttingEnds = {}
	endCut = -1

	#first pass to determine segments to be discarded
	for row in oldCff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if (row[2] == 'boundary' or row[2] == 'boundary_sync') and simpleSeg.orth != None and row[1] > endCut:
			if simpleSeg.getNormalizedConf() != None:
				if simpleSeg.getNormalizedConf() <= threshold: #replace if acoustic scores are used as confidence measure
				#if simpleSeg.getNormalizedConf() > threshold: #replacement
					cuttingPoints[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
					cuttingEnds[simpleSeg.end] = None
				else:
					pass
			else:
				print 'warning: no confidence score provided for ' + str(row)
			simpleSeg.reset()
		if row[2] == 'orth':
			simpleSeg.appendOrth(row[3])
		if row[2] == 'score':
			confDict = row[3]
			simpleSeg.addConf(confDict['confidence'])

	# second pass
	for row in oldCff.getRows():
		if (row[2] == 'boundary' or row[2] == 'boundary_sync'):
			if (row[0],row[1]) in cuttingPoints.keys():
				endCut = cuttingPoints[(row[0],row[1])]
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				if row[1] in cuttingEnds.keys():
					#pass
					newCff.setRow(row[0],row[1],row[2],row[3])
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
		elif row[1] < endCut:
			pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	if isStatistics:
		simpleFilterStats(cuttingPoints, oldCff)
	return removeEmptySegs(newCff)


#replace words with a confScore lower than threshold with [IGNORE]; if a lexicon is provided, n [IGNORE] tokens are inserted (n= # phonemes of the original word)
def replaceWordsConfidence(cff, threshold, lexicon=None, isStatistics=False):
	oldCff = insertSync(cff)
	if lexicon != None:
		orthDict = lexiconList2Dict(lexiconParser(lexicon, orthList = []))
	else:
		orthDict = {}
	newCff = Cff()
	simpleSeg = SimpleSeg()
	ignoreDict = {}

	for row in oldCff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if (row[2] == 'boundary' or row[2] == 'boundary_sync') and simpleSeg.orth != None:
			if simpleSeg.getNormalizedConf() != None:
				newCff.setRow(row[0],row[1],row[2],row[3])
				if (simpleSeg.getNormalizedConf() < threshold):
					ignoreDict[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
					#newCff.setRow(simpleSeg.recording, simpleSeg.start ,'orth','[IGNORE]')
					currentOrth = re.sub(r' +', r' ', simpleSeg.getOrth().upper())
					currentOrth = re.sub(r'^ ', r'', currentOrth)
					currentOrth = re.sub(r' $', r'', currentOrth)
					if orthDict.has_key(currentOrth):
						tempList = orthDict[currentOrth][0].split()
						tempString = None
						for i in range(len(tempList)):
							if tempString == None:
								tempString = '[IGNORE]'
							else:
								tempString += ' [IGNORE]'
						newCff.setRow(simpleSeg.recording, simpleSeg.start ,'orth', tempString)
					else:
						newCff.setRow(simpleSeg.recording, simpleSeg.start ,'orth','[IGNORE]')
						#print 'warning: orth %s not in lexicon' % currentOrth
				else:
					newCff.setRow(simpleSeg.recording, simpleSeg.start, 'orth', simpleSeg.getOrth())
			else:
				print 'warning: no confidence score provided for ' + str(row)
			simpleSeg.reset()

		elif row[2] == 'score':
			newCff.setRow(row[0],row[1],row[2],row[3])
			confDict = row[3]
			simpleSeg.addConf(confDict['confidence'])

		elif row[2] == 'orth':
			simpleSeg.appendOrth(row[3])

		else:
			newCff.setRow(row[0],row[1],row[2],row[3])

	if isStatistics:
		crosswordViolations(newCff)
		simpleFilterStats(ignoreDict, oldCff)
	return removeEmptySegs(removeSync(newCff))

def insertCrosswordContext(cff):
	oldCff = cff
	newCff = Cff()
	previousOrth = None

	for row in oldCff.getRows():

		if (row[2] == 'boundary' or row[2] == 'boundary_sync'):# and statistics.orth != None:
			if row[2] == 'boundary':
				previousOrth = None
			newCff.setRow(row[0], row[1], row[2], row[3])
		if row[2] == 'orth':
			if previousOrth == None:
				previousOrth = row[3]
				newCff.setRow(row[0], row[1], row[2], row[3])
			elif previousOrth != None and (previousOrth == '[IGNORE]' or previousOrth.find('[') == -1) :
				if (row[3] == '[IGNORE]' or row[3].find('[') == -1) :
					if (previousOrth == '[IGNORE]' and row[3] != '[IGNORE]') or (previousOrth != '[IGNORE]' and row[3] == '[IGNORE]'):
						newCff.setRow(row[0], row[1], row[2], '[CONTEXT] ' + row[3])
					else:
						newCff.setRow(row[0], row[1], row[2], row[3])
				else:
					newCff.setRow(row[0], row[1], row[2], row[3])
			else:
				newCff.setRow(row[0], row[1], row[2], row[3])
			previousOrth = row[3]
		else:
			newCff.setRow(row[0], row[1], row[2], row[3])
	return newCff

# replace words with a confScore lower than threshold with [IGNORE]
# def replaceWordsConfidence(cff, threshold, isStatistics=False):
# 	oldCff = insertSync(cff)
# 	newCff = Cff()
# 	simpleSeg = SimpleSeg()
# 	ignoreDict = {}
# 	ignoreEnds = {}
# 	isIgnorable = False

# 	for row in oldCff.getRows():
# 		simpleSeg.setKey(row[0], row[1])

# 		if (row[2] == 'boundary' or row[2] == 'boundary_sync') and simpleSeg.orth != None:
# 			if simpleSeg.getNormalizedConf() != None:
# 				if simpleSeg.getNormalizedConf() < threshold: #replace if acoustic scores are used as confidence measure
# 				#if simpleSeg.getNormalizedConf() > threshold: #replacement
# 					ignoreDict[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
# 					ignoreEnds[simpleSeg.end] = None
# 				else:
# 					pass
# 			else:
# 				print 'warning: no confidence score provided for ' + str(row)
# 			simpleSeg.reset()
# 		if row[2] == 'orth':
# 			simpleSeg.appendOrth(row[3])
# 		if row[2] == 'confidence':
# 			confDict = row[3]
# 			simpleSeg.addConf(confDict['score'])

# 	# second pass
# 	for row in oldCff.getRows():
# 		if (row[2] == 'boundary' or row[2] == 'boundary_sync'):
# 			if (row[0],row[1]) in ignoreDict.keys():
# 				isIgnorable = True
# 			else:
# 				isIgnorable = False
# 			newCff.setRow(row[0],row[1],row[2],row[3])

# 		elif row[2] == 'orth':
# 			if isIgnorable:
# 				newCff.setRow(row[0], row[1], row[2], '[IGNORE]')
# 			else:
# 				newCff.setRow(row[0],row[1],row[2],row[3])
# 		else:
# 			newCff.setRow(row[0],row[1],row[2],row[3])
# 	if isStatistics:
# 		crosswordViolations(newCff)
# 		simpleFilterStats(ignoreDict, oldCff)
# 	return removeEmptySegs(removeSync(newCff))


# assign the segmentwise normalized WPP to every word of the segment
def assignNormalizedWPP(cff):
	oldCff = cff
	newCff = Cff()
	simpleSeg = SimpleSeg ()
	normalizedWPPs = {}
	isSpeech = False
	currentWPP = None

	# first pass to determine normalized WPPs
	for row in oldCff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if row[2] == 'boundary' and simpleSeg.orth != None and row[1]:
			if simpleSeg.getNormalizedConf() != None:
				normalizedWPPs[(simpleSeg.recording, simpleSeg.start)] = round(simpleSeg.getNormalizedConf(), 6)
			else:
				print 'warning: no confidence score provided for ' + str(row)
			simpleSeg.reset()
		if row[2] == 'orth':
			simpleSeg.appendOrth(row[3])
			if row[3].startswith('['):
				isSpeech = False
			else:
				isSpeech = True
		if row[2] == 'score':
			confDict = row[3]
			if isSpeech == True:
				simpleSeg.addConf(confDict['confidence'])
			else:
				pass

	# second pass
	for row in oldCff.getRows():
		if row[2] == 'boundary':
			if (row[0],row[1]) in normalizedWPPs.keys():
				currentWPP = normalizedWPPs[(row[0],row[1])]
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				currentWPP = None
				newCff.setRow(row[0],row[1],row[2],row[3])
		elif row[2] == 'score':
			confDict = row[3]
			if currentWPP != None:
				confDict['confidence'] = round(currentWPP, 6)
			newCff.setRow(row[0],row[1],row[2],confDict)
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	return newCff

# working with ARPA LMs
def read_arpaLM(lmFile, order):
	isPickle = False
	firstNgram = True
	counter = 0
	percent = 0.0
	print "opening ARPA LM file..."
	if lmFile.endswith('.gz'):
		lmFD = gzip.open(lmFile, 'r')
		if lmFile.find('pickle') != -1:
			isPickle = True
	else:
		lmFD = codecs.open(lmFile, 'r')
		if lmFile.endswith('pickle'):
			isPickle = True
	if isPickle:
		print "reading pickled (ARPA LM) ngram prefix tree..."
		pTree = cPickle.load(lmFD)
		lmFD.close()
		return pTree
	else:
		print "reading ARPA LM..."
		lines = lmFD.readlines()
		for line in lines:
			if line.startswith('ngram'):
				temp = line.split(' ')
				ngramType = temp[1][:temp[1].find('=')]
				if ngramType == str(order):
					totalNgrams = line[line.find('=')+1:line.find('\n')]
					print "found %s %s-grams..." % (totalNgrams, order)
			elif line.startswith('\data\\') or line.startswith('\end\\'):
				pass
			elif line.startswith('\\') and line[1].isdigit():
				currentNgramType = line[1:line.find('-')]
				if str(currentNgramType) == str(order):
					print "reading " +currentNgramType+ "-grams..."
			elif line.startswith('\n'):
				pass
			elif str(currentNgramType) == str(order):
				counter+=1
				oldPercent = 0.0
				percent = float(getPercent(counter, totalNgrams))
				if round(percent) > round(oldPercent):
					print "done: %s%%   \r" % (percent),
				temp = line.split('\t')
				if len(temp) < 2:
					print "Error in LM-file: %s has too few fields!" % temp
				if temp[1].endswith('\n'):
					currentNgram = temp[1][:temp[1].find('\n')]
				else:
					currentNgram = temp[1]
				if order > 1:
					ngramList = currentNgram.split(' ')
				else:
					ngramList = currentNgram
				if firstNgram == True:
					pTree = Node(ngramList)
					firstNgram = False
				else:
					pTree.add_tail(ngramList)
		lmFD.close()
		print "done!"
		return pTree

def write_arpaLMtree(destFile, pTree):
	print "writing pickled ARPA LM ngram list..."
	dumpFile = codecs.open(destFile, 'w')
	cPickle.dump(pTree, dumpFile)
	dumpFile.close()
	print "done!"

class NgramSeg:
	def __init__(self):
		self.isNewRecording = True
		self.prefixtree = None
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
			self.isNewRecording = True
		elif self.recording != recording:
			self.recording = recording
			self.isNewRecording = True
		elif self.recording == recording:
			self.isNewRecording = False
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

	def setPrefixtree(self, tree):
		self.prefixtree = tree

	def getCoverage(self, orth, order):
		nGram=[]
		orth = re.sub("\[\S+\]", "", orth)
		orth = re.sub(" +", " ", orth)
		orth = re.sub("^ ", "", orth)
		orth = re.sub(" $", "", orth)
		orthList = orth.split(' ')
		orthEvalList = []
		totalWords = 0
		totalMatches = 0
		if len(orthList) >= order:
			for word in orthList:
				orthEvalList += [[word, 0]]
			for item in range(0,len(orthList)-(order-1)):
				for i in range(0,order):
					if nGram == []:
						nGram = [orthList[item+i]]
					else:
						nGram += [orthList[item+i]]
				if self.prefixtree.isMember(nGram):
					for i in range(0,order):
						orthEvalList[item+i][1] = 1
				else:
					pass
				nGram = []
			for word in orthEvalList:
				totalWords += 1
				if word[1] == 1:
					totalMatches +=1
			return str(round(float(totalMatches)/float(totalWords),2))
		else:
			return(0.0)

# simple realisation of a prefix tree for FTE and LM
class Node:
	def __init__(self, nGramList):
		self.tails = {}
		self.eos = 0  # end of list
		self.maxlen = 0
		self.add_tail(nGramList)

	def add_tail(self, nGramList):
		if len(nGramList) > self.maxlen:
			self.maxlen = len(nGramList)
		if len(nGramList) == 0:
			self.eos = 1
			return
		if self.tails.has_key(nGramList[0]):
			self.tails[nGramList[0]].add_tail(nGramList[1:])
		else:
			self.tails[nGramList[0]] = self.__class__(nGramList[1:])

	def isMember(self, nGramList):
		#print 'nGram List: ' + str(nGramList) #
		words = []
		rests = []
		for word, rest in self.tails.items():
			words += [word]
		if len(nGramList) == 0:
			#print "list was empty"#
			self.eos = 1
			return True
		if nGramList[0] not in words:
			return False
		elif nGramList[0] in words:
			#print 'word: ' + str(nGramList[0])#
			#print 'keys on this layer: ' + str(words)#
			#print 'matching tree: ' + str(self.tails[nGramList[0]]) + '\n'#
			if len(nGramList[1:]) == 0:
				return True
			elif len(nGramList[1:]) != 0:
				#print len(nGramList[1:])#
				rest = self.tails[nGramList[0]]
				result = rest.isMember(nGramList[1:])
				return result

	def printTree(self):
		words = []
		for word, rest in self.tails.items():
			words += [word]
		if words != []:
			print words
		for word in words:
			if self.tails[word] != None:
				self.tails[word].printTree()

def prefixtree(nGrams):
	tree = Node(nGrams[0])
	for ngram in nGrams[1:]:
		tree.add_tail(ngram)
	return tree

# removal of segments which have a ngram-coverage lower than threshold
def removeSegsWithLm(cff, lm, order=3, threshold=0.4, writeLmDestination = None, isStatistics = False):
	nGramDict = {}
	newCff = Cff()
	nGramSeg = NgramSeg ()
	cuttingPoints = {}
	cuttingEnds = {}
	endCut = -1

	pTree = read_arpaLM(lm, order)
	nGramSeg.setPrefixtree(pTree)
	if writeLmDestination != None:
		write_arpaLMtree(writeLmDestination, pTree)

	#first pass to determine segments to be discarded
	for row in cff.getRows():
		nGramSeg.setKey(row[0], row[1])

		if row[2] == 'boundary' and nGramSeg.orth != None and row[1] > endCut:
			if nGramSeg.getCoverage(nGramSeg.orth, order) < threshold:
				cuttingPoints[(nGramSeg.recording, nGramSeg.start)] = nGramSeg.end
				cuttingEnds[nGramSeg.end] = None
			else:
				pass
			nGramSeg.reset()

		if row[2] == 'orth':
			nGramSeg.appendOrth(row[3])

	# second pass
	for row in cff.getRows():
		if row[2] == 'boundary':
			if (row[0],row[1]) in cuttingPoints.keys():
				endCut = cuttingPoints[(row[0],row[1])]
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				if row[1] in cuttingEnds.keys():
					newCff.setRow(row[0],row[1],row[2],row[3])
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
		elif row[1] < endCut:
			pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	if isStatistics:
		simpleFilterStats(cuttingPoints, cff)
	return removeEmptySegs(newCff)

# working with (FTE) Text Files
def read_text(textFile, order):
	isPickle = False
	firstNgram = True
	nGram = []
	print "opening text file..."
	if textFile.endswith('.gz'):
		textFD = gzip.open(textFile, 'r')
		if textFile.find('pickle') != -1:
			isPickle = True
	else:
		textFD = codecs.open(textFile, 'r')
		if textFile.endswith('pickle'):
			isPickle = True
	if isPickle:
		print "reading pickled ngram prefix tree..."
		pTree = cPickle.load(textFD)
		textFD.close()
		return pTree
	else:
		print "reading text file..."
		lines = textFD.readlines()
		for line in lines:
			orth = re.sub("\[\S+\]", "", line.decode("UTF-8"))
			orth = re.sub(" s ","'s ", orth)
			orth = re.sub("_"," ", orth)
			orth = re.sub(" +", " ", orth)
			orth = re.sub("^ ", "", orth)
			orth = re.sub(" $", "", orth)
			if orth.startswith('<s>'):
				orth=orth[4:-6]
			else:
				pass
			orthList = orth.split(' ')
			if len(orthList) >= order:
				for item in range(0,len(orthList)-(order-1)):
					for i in range(0,order):
						if nGram == []:
							nGram = [orthList[item+i]]
						else:
							nGram += [orthList[item+i]]
					if firstNgram == True:
						pTree = Node(nGram)
						firstNgram = False
					else:
						pTree.add_tail(nGram)
					nGram = []

		textFD.close()
		print "done!"
		return pTree

def write_textFileTree(destFile, pTree):
	print "writing pickled text file ngram tree..."
	dumpFile = codecs.open(destFile, 'w')
	cPickle.dump(pTree, dumpFile)
	dumpFile.close()
	print "done!"


# removal of segments which have a ngram-coverage lower than threshold
def removeSegsWithTextFile(cff, text, order=3, threshold=0.5, writeLmDestination = None, isStatistics = False):
	nGramDict = {}
	newCff = Cff()
	nGramSeg = NgramSeg ()
	cuttingPoints = {}
	cuttingEnds = {}
	endCut = -1

	pTree = read_text(text, order)
	nGramSeg.setPrefixtree(pTree)
	if writeLmDestination != None:
		write_textFileTree(writeLmDestination, pTree)

	#first pass to determine segments to be discarded
	for row in cff.getRows():
		nGramSeg.setKey(row[0], row[1])

		if row[2] == 'boundary' and nGramSeg.orth != None and row[1] > endCut:
			if float(nGramSeg.getCoverage(nGramSeg.orth, order)) < float(threshold):
				cuttingPoints[(nGramSeg.recording, nGramSeg.start)] = nGramSeg.end
				cuttingEnds[nGramSeg.end] = None
			else:
				pass
			nGramSeg.reset()

		if row[2] == 'orth':
			nGramSeg.appendOrth(row[3])

	# second pass
	for row in cff.getRows():
		if row[2] == 'boundary':
			if (row[0],row[1]) in cuttingPoints.keys():
				endCut = cuttingPoints[(row[0],row[1])]
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				if row[1] in cuttingEnds.keys():
					newCff.setRow(row[0],row[1],row[2],row[3])
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
		elif row[1] < endCut:
			pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	if isStatistics:
		simpleFilterStats(cuttingPoints, cff)
	return removeEmptySegs(newCff)


# removal of all noises in the bliss noise format
def removeNoises(cff, isStatistics = False):

	def makeNoiseless(orth):
		orth = re.sub("\[\S+\]", "", orth)
		orth = re.sub(" +", " ", orth)
		orth = re.sub("^ ", "", orth)
		orth = re.sub(" $", "", orth)
		orth = re.sub(" (uh )+", " ", orth)
		orth = re.sub(" +", " ", orth)
		orth = re.sub("^ ", "", orth)
		orth = re.sub(" $", "", orth)
		orth = re.sub("^uh ", "", orth)
		orth = re.sub(" uh$", "", orth)
		orth = re.sub("^uh$", "", orth)
		return orth

	syncCff = Cff()
	#pat = re.compile('\[[^ \[\]]*\]')

	for row in cff.getRows():
		if row[2] == 'orth':
			if row[3] != '':
				newOrth = makeNoiseless(row[3])
				#if newOrth == '':
				#	newOrth = 'TOBEREMOVED'
				syncCff.setRow(row[0],row[1],row[2],newOrth)
			else:
				syncCff.setRow(row[0],row[1],row[2])
		else:
			syncCff.setRow(row[0],row[1],row[2],row[3])
	return syncCff

# removal of all segments included in the dictionary-style file dictFile
def removeWithDictFile(cff, dictFile, isStatistics = False):
	newCff = Cff()
	dictFD = codecs.open(dictFile, 'r')
	cuttingPoints = {}
	cuttingEnds = {}
	endCut = -1

	# filling of the dictionary
	for line in dictFD:
		entry = line[:-1].split(' ')
		cuttingPoints[(entry[0], str(round(float(entry[1]), 3)))] = entry[2]
		cuttingEnds[str(round(float(entry[2]), 3))] = None
	# discarding of the segments
	for row in cff.getRows():
		if row[2] == 'boundary':
			if (row[0],str(round(float(row[1]), 3))) in cuttingPoints.keys():
				endCut = cuttingPoints[(row[0],str(round(float(row[1]), 3)))]
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				if str(round(float(row[1]), 3)) in cuttingEnds.keys():
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					#endCut = -1.0
		elif row[1] < endCut:
			pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	if isStatistics:
		simpleFilterStats(cuttingPoints, cff)
	return removeEmptySegs(newCff)

# longest possible merge of segments
def mergeSegments(cff):
	mergeCff = Cff()
	rows = cff.getRows()
	isBoundary = False
	maximum = len(rows)-1
	for row in range(len(rows)):
		if rows[row][2] == 'boundary':
			if row == 0 or row == maximum:
				mergeCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
			elif row < maximum and rows[row+1][2] == 'boundary':
				isBoundary = True
				mergeCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
			elif isBoundary == True:
				isBoundary = False
				mergeCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
			else:
				pass
		else:
			mergeCff.setRow(rows[row][0], rows[row][1], rows[row][2], rows[row][3])
			isBoundary = False
	return mergeCff

def removeOrthByScore(cff, scoreName, lowerBound, upperBound):
	newCff = Cff()
	preTime = None
	preRowList = []
	removeOrth = False
	for row in cff.getRows():
		preRowList.append(row)

		if row[2] == 'score':
			curScore = float(row[3][scoreName])
			if curScore > upperBound or curScore < lowerBound:
				removeOrth = True
			else:
				removeOrth = False

		if preTime != float(row[1]):
			for addRow in preRowList:
				if addRow[2] != 'orth' or removeOrth == False:
					newCff.setRow(addRow[0],addRow[1],addRow[2],addRow[3])
			preRowList = []
			preTime = float(row[1])

	for addRow in preRowList:
		newCff.setRow(addRow[0],addRow[1],addRow[2],addRow[3])

	return newCff


def escapeOrthForXml(cff):
	newCff = Cff()
	for row in cff.getRows():
		if row[2] == 'orth':
			orth=row[3].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
			newCff.setRow(row[0],row[1],row[2],orth)
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])

	return newCff

def removeOrthIfLonger(cff, length):
	newCff = Cff()
	preBoundary = None
	preRowList = []
	for row in cff.getRows():
		preRowList.append(row)

		if row[2] == 'boundary':
			curBoundary = float(row[1])
			if preBoundary != None and (curBoundary - preBoundary) > length:
				removeOrth = True
			else:
				removeOrth = False

			preBoundary = float(row[1])

			for addRow in preRowList:
				if addRow[2] != 'orth' or removeOrth == False:
					newCff.setRow(addRow[0],addRow[1],addRow[2],addRow[3])
			preRowList = []

	for addRow in preRowList:
		newCff.setRow(addRow[0],addRow[1],addRow[2],addRow[3])

	return newCff

def removeOrthIfShorter(cff, length):
	newCff = Cff()
	preBoundary = None
	preRowList = []
	for row in cff.getRows():
		preRowList.append(row)

		if row[2] == 'boundary':
			curBoundary = float(row[1])
			if preBoundary != None and (curBoundary - preBoundary) < length:
				removeOrth = True
			else:
				removeOrth = False

			preBoundary = float(row[1])

			for addRow in preRowList:
				if addRow[2] != 'orth' or removeOrth == False:
					newCff.setRow(addRow[0],addRow[1],addRow[2],addRow[3])
			preRowList = []

	for addRow in preRowList:
		newCff.setRow(addRow[0],addRow[1],addRow[2],addRow[3])

	return newCff

# working with Lexica
class lexiconhandler(ContentHandler):
	def __init__(self, plist=[]):
		self.orthList = plist
		self.orthDict = {}
		self.isLemma = 0
		self.isOrth = 0
		self.isPhon = 0
		self.phonemes = ''
		self.orthTmp = ''
		self.orthTmpList = []
		self.phonTmpList = []
		self.orthPos = 0
		self.orthInList = 0

	def startElement(self, name, attrs):
		if name=='lemma':
			self.isLemma = 1
		elif name=='orth':
			self.isOrth = 1
		elif name =='phon':
			self.isPhon = 1
		else:
			pass

	def endElement(self, name):
		if name=='lemma':
			if self.orthTmpList != []:
				for item in self.orthTmpList:
					if self.orthDict.has_key(item):
						self.orthInList = 1
						self.orthPos = self.orthDict[item]
					else:
						pass
				if self.orthInList == 0:
					self.orthList.append([self.orthTmpList,self.phonTmpList])
					for item in self.orthTmpList:
						self.orthDict[item]= len(self.orthList)-1
				elif self.orthInList == 1:
					for item in self.orthTmpList:
						if item not in self.orthList[self.orthPos][0]:
							self.orthList[self.orthPos][0].append(item)
					for item in self.phonTmpList:
						if item not in self.orthList[self.orthPos][1]:
							self.orthList[self.orthPos][1].append(item)
			self.isLemma = 0
			self.orthInList = 0
			self.orthTmpList = []
			self.phonTmpList = []
		elif name=='orth':
			if self.isLemma == 1:
				self.orthTmpList += [self.orthTmp.upper()]
			self.orthTmp = ''
			self.isOrth = 0
		elif name=='phon':
			if self.isLemma == 1:
				self.phonTmpList += [self.phonemes]
			self.phonemes = ''
			self.isPhon = 0

	def characters(self,ch):
		if self.isLemma == 1 and self.isOrth == 1:
			self.orthTmp += ch
		elif self.isLemma == 1 and self.isPhon == 1:
			#self.phonemes += str(ch).lstrip()
			self.phonemes += str(ch).replace('\n', '')

def lexiconParser(inFileName, orthList = []):
	parser=make_parser()
	handler=lexiconhandler(orthList)
	parser.setContentHandler( handler )
	parser.parse(open(inFileName, "rb") )
	return handler.orthList

def lexiconList2Dict(orthList):
	phonDict = {}
	for item in orthList:
		for orth in item[0]:
			phonDict[orth]=[]
			for phon in item[1]:
				phonDict[orth].append(phon)
	return phonDict

# convert orthography to phoneme sequence using lexicon lex and the respective first pronunciation variant
def orth2phon(cff, lex):
	newCff = Cff()
	tmpOrth = []
	newOrth = []
	skipNextItem = False
	orthDict = lexiconList2Dict(lexiconParser(lex, orthList = []))

	for row in cff.getRows():
		if row[2] == 'orth' and row[3] != None:
			orth = row[3]
			orth = re.sub(" +", " ", orth)
			orth = re.sub("^ ", "", orth)
			orth = re.sub(" $", "", orth)
			#orth = re.sub(r"([A-Z])\.", r"\1", orth)
			tmpOrth = orth.split(' ')
			for item in range(0,len(tmpOrth)):
				if skipNextItem == True:
					skipNextItem = False
				elif orthDict.has_key(tmpOrth[item]):
					newOrth.append(orthDict[tmpOrth[item]][0])
				elif orthDict.has_key(' '.join(tmpOrth[item-1:item+1])):
					del newOrth[item-1]
					newOrth.append(orthDict[' '.join(tmpOrth[item-1:item+1])][0])
				elif orthDict.has_key(' '.join(tmpOrth[item-2:item+1])):
					del newOrth[item-2]
					del newOrth[item-1]
					newOrth.append(orthDict[' '.join(tmpOrth[item-2:item+1])][0])
				elif item < len(tmpOrth)-2:
					if orthDict.has_key(' '.join(tmpOrth[item-1:item+2])):
						del newOrth[item-1]
						newOrth.append(orthDict[' '.join(tmpOrth[item-1:item+2])][0])
						skipNextItem = True
					else:
						if not tmpOrth[item].startswith('('):
							print "entity not found in lexicon: ", tmpOrth[item].encode('UTF-8')
				else:
					if not tmpOrth[item].startswith('('):
						print "entity not found in lexicon: ", tmpOrth[item].encode('UTF-8')
			newCff.setRow(row[0], row[1], row[2], ' '.join(newOrth))
			tmpOrth = []
			newOrth = []
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	return (newCff)


def keepOrthByVocabulary(cff, knownWordsList):
	newCff = Cff()

	for row in cff.getRows():
		if row[2] == 'orth':
			removeOrth = False
			for word in row[3].split(' '):
				if word not in knownWordsList:
					removeOrth = True

			if removeOrth == False:
				newCff.setRow(row[0],row[1],row[2],row[3])
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])

	return newCff

# distribute orth linearly over the segment (important for NIST-scoring: .ctm-files)
def distributeOrth(cff):
	newCff = Cff()
	cffInfo = SimpleSeg()

	for row in cff.getRows():
		cffInfo.setKey(row[0], row[1])

		if row[2] == 'boundary' and cffInfo.orth != None:
			tmpOrth = re.sub(" +", " ", cffInfo.orth)
			tmpOrth = re.sub("^ ", "", tmpOrth)
			tmpOrth = re.sub(" $", "", tmpOrth)
			orthList = tmpOrth.split(' ')
			for word in range(len(orthList)):
				newStart = float(cffInfo.start) + float(word/float(len(orthList))) * float(cffInfo.end - cffInfo.start)
				#print word, cffInfo.start, cffInfo.end, newStart
				newCff.setRow(row[0], newStart, "orth", orthList[word])
				newCff.setRow(row[0], newStart, "boundary")
				if cffInfo.conf != None:
					confDict = { u'score' : cffInfo.conf }
					newCff.setRow(row[0], newStart, "confidence", confDict)
				newStart = 0.0
			newCff.setRow(row[0], row[1], row[2])
			cffInfo.reset()
			newStart = 0.0
		if row[2] == 'orth':
			cffInfo.appendOrth(row[3])
		if row[2] == 'confidence':
			confDict = row[3]
			cffInfo.setConf(confDict['score'])
		else:
			newCff.setRow(row[0], row[1], row[2], row[3])
	return newCff

# remove all segments which do not contain any regular lemma
def removeNoiseSegments(cff, isStatistics):
	newCff = Cff()
	simpleSeg = SimpleSeg()
	cuttingPoints = {}
	cuttingEnds = {}
	endCut = -1

	def makeNoiseless(orth):
		orth = re.sub("\[\S+\]", "", orth)
		orth = re.sub(" +", " ", orth)
		orth = re.sub("^ ", "", orth)
		orth = re.sub(" $", "", orth)
		orth = re.sub(" (uh )+", " ", orth)
		orth = re.sub(" +", " ", orth)
		orth = re.sub("^ ", "", orth)
		orth = re.sub(" $", "", orth)
		orth = re.sub("^uh ", "", orth)
		orth = re.sub(" uh$", "", orth)
		orth = re.sub("^uh$", "", orth)
		if orth == '':
			return None
		else:
			return orth

	#first pass to determine segments to be discarded
	for row in cff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if row[2] == 'boundary' and simpleSeg.orth != None:
			if makeNoiseless(simpleSeg.orth) == None:
				cuttingPoints[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
				cuttingEnds[simpleSeg.end] = None
			else:
				pass
			simpleSeg.reset()

		if row[2] == 'orth':
			simpleSeg.appendOrth(row[3])

	# second pass
	for row in cff.getRows():
		if row[2] == 'boundary':
			if (row[0],row[1]) in cuttingPoints.keys():
				endCut = cuttingPoints[(row[0],row[1])]
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				if row[1] in cuttingEnds.keys():
					newCff.setRow(row[0],row[1],row[2],row[3])
				else:
					newCff.setRow(row[0],row[1],row[2],row[3])
					endCut = -1.0
		elif row[1] < endCut:
			pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])
	if isStatistics:
		simpleFilterStats(cuttingPoints, cff)
	return removeEmptySegs(newCff)

def getSgmlData(inFileName):
	sgmlDict = {}
	fd = codecs.open(inFileName, 'r')
	lines = fd.readlines()
	isPathline = False
	for line in lines:
		line = line[:-1]
		if isPathline == True:
			isPathline = False
			if len(line) > 0:
				items = line.split(':')
				for item in items:
					words = item.split(',')
					levOp = words[0]
					ref = words[1]
					ref = re.sub(r'^"', '', ref)
					ref = re.sub(r'"$', '', ref)
					if ref == '':
						ref = None
					hyp = words[2]
					hyp = re.sub(r'^"', '', hyp)
					hyp = re.sub(r'"$', '', hyp)

					if levOp == 'C' and ref=='(%hesitation)':
						startTime = None
						endTime = None
						hyp = None
					elif levOp != 'D':
						times = words[3].split('+')
						startTime = times[0]
						endTime = times[1]
					else:
						startTime = None
						endTime = None
						hyp = None
					#print levOp, ref, hyp, startTime, endTime
					if sgmlDict.has_key((currentFile, round(float(segmentStart), 3))):
						sgmlDict[(currentFile, round(float(segmentStart), 3))].append([currentSpeaker, segmentEnd, levOp, ref, hyp, startTime, endTime])
					else:
						sgmlDict[(currentFile, round(float(segmentStart), 3))]=[]
						sgmlDict[(currentFile, round(float(segmentStart), 3))]=[[currentSpeaker, segmentEnd, levOp, ref, hyp, startTime, endTime]]

		elif line.startswith('<PATH'):
			isPathline = True
			currentFile = line[line.find('file="')+6:line.find('channel')-2].upper()
			currentSpeaker = line[line.find('id="')+4:line.find('word_cnt')-2][1:-1]
			segmentStart = line[line.find('R_T1="')+6:line.find('R_T2="')-2]
			segmentEnd = line[line.find('R_T2="')+6:line.find('word_aux="')-2]
			#print currentFile, currentSpeaker, segmentStart, segmentEnd

	return sgmlDict

def addSgmlToCff(cff, inFileName):
	oldCff = cff
	newCff = Cff()
	sgmlDict = getSgmlData(inFileName)
	sgmlList = []
	keys = sgmlDict.keys()
	keys.sort()
	deletions = ''
	rows = oldCff.getRows()


	for row in range(len(rows)):

		if rows[row][2] == 'boundary'  and (row < (len(rows)-1) and rows[row+1][2] != 'boundary'):
			sgmlList = sgmlDict[keys[0]]
			keys.remove(keys[0])
			newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])


		if rows[row][2] == 'orth' and rows[row][3].find('[') == -1:
			data = None
			item = sgmlList[0]
			if item[5] == None:
				while item[5] == None:
					if item[3] == '(%hesitation)':
						sgmlList.remove(item)
						item = sgmlList[0]
					else:
						deletions += '<Comment desc="Ref: '+str(item[3]).decode('latin-1')+' ('+str(item[2])+')"/>'
						sgmlList.remove(item)
						item = sgmlList[0]
			if deletions == '' and item[2] == 'C':
				#data = '<Comment desc="Ref: '+str(item[3]).decode('latin-1')+' ('+str(item[2])+')"/>'+' '+rows[row][3]
				data = rows[row][3]
			else:
				data = deletions+'<Comment desc="Ref: '+str(item[3]).decode('latin-1')+' ('+str(item[2])+')"/>'+' '+rows[row][3]
				deletions = ''
			sgmlList.remove(item)

			if data != None:
				newCff.setRow(rows[row][0],rows[row][1],rows[row][2],data)
			else:
				newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])
		else:
			newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])

	return newCff

def sentenceBoundaries2segmentBoundaries(cff, context=True):
	noContext = ['[SILENCE]', '[PAUSE]', '[NOISE]', '[ARTIC]', '[B]', '[HESITATION]', 'boundary']
	oldCff = cff
	newCff = Cff()
	newCff2 = Cff()
	rows = oldCff.getRows()
	sentBoundary = None
	nextTime = None
	nextOrth = None
	prevTime = None
	prevOrth = None
	counter = 0
	deleteDict = {}

	for row in range(len(rows)):
		if rows[row][2] == 'orth':
			if rows[row][3].strip() in ['[SENTENCE-END]', '[SENTENCE-BEGIN]']:
				deleteDict[(rows[row][0], rows[row][1])] = None

	if context:
		for row in range(len(rows)):
			if rows[row][2] == 'orth':
				if rows[row][3].strip() == '[SENTENCE-END]':
					try:
						counter = row
						while (round(float(rows[counter][1]), 3) == round(float(rows[row][1]), 3) or rows[counter][2] != 'orth'):
							counter+=1
						else:
							nextOrth = rows[counter][3]
							nextTime = rows[counter][1]
					except:
						nextTime = None
						nextOrth = None
						print "no time / orthography after Sentence Boundary found"

					try:
						counter = row-1
						while rows[counter][2] != 'orth':
							counter-=1
						else:
							prevOrth = rows[counter][3]
							prevTime = rows[counter][1]
					except:
						nextTime = None
						nextOrth = None
						print "no time / orthography  before Sentence Boundary found"
					if (nextTime != None and prevTime != None):
						#sentBoundary = str(round(float(rows[row][1]) + (float(nextTime) - round(float(rows[row][1]), 3))/2, 3))
						#print prevOrth, prevTime, nextOrth, nextTime
						if (prevOrth.strip() in noContext) or (nextOrth.strip() in noContext):
							sentBoundary = str(round(float(nextTime), 3))
							newCff.setRow(rows[row][0],sentBoundary,'boundary')
						else:
							pass
				elif rows[row][3].strip() == '[SENTENCE-BEGIN]':
					pass
				else:
					newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])
			elif (rows[row][0], rows[row][1]) in deleteDict:
				pass
			else:
				newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])


	else:
		for row in range(len(rows)):
			if rows[row][2] == 'orth':
				if rows[row][3].strip() == '[SENTENCE-END]':
					try:
						counter = row
						while round(float(rows[counter][1]), 3) == round(float(rows[row][1]), 3):
							counter+=1
						else:
							nextTime = rows[counter][1]
					except:
						nextTime = None
						print "no time after Sentence Boundary found"
					#sentBoundary = str(round(float(rows[row][1]) + (float(nextTime) - round(float(rows[row][1]), 3))/2, 3))
					sentBoundary = str(round(float(nextTime), 3))
					newCff.setRow(rows[row][0],sentBoundary,'boundary')
				elif rows[row][3].strip() == '[SENTENCE-BEGIN]':
					pass
				else:
					newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])
			elif (rows[row][0], rows[row][1]) in deleteDict:
				pass
			else:
				newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])

	return newCff

def removeSentenceBoundaries(cff):
	oldCff = cff
	newCff = Cff()
	deleteDict = {}

	rows = oldCff.getRows()

	for row in range(len(rows)):
		if rows[row][2] == 'orth':
			if rows[row][3].strip() in ['[SENTENCE-END]', '[SENTENCE-BEGIN]']:
				deleteDict[(rows[row][0], rows[row][1])] = None

	for row in range(len(rows)):
		if (rows[row][0], rows[row][1]) in deleteDict:
			pass
		else:
			newCff.setRow(rows[row][0],rows[row][1],rows[row][2],rows[row][3])

	return newCff

# joins noisy events in a segment to [SILENCE]

#remove rows by timepoints via dictionary and set appropriate boundaries
def removeWithDictFileTimepoints(cff, dictFile, isStatistics = False):
	newCff = Cff()
	dictFD = codecs.open(dictFile, 'r')
	cuttingPoints = {}
	recordingDict = {}

	# filling of the dictionary
	for line in dictFD:
		entry = line[:-1].split(' ')
		cuttingPoints[(entry[0], str(round(float(entry[1]), 3)))] = entry[2]
		recordingDict[entry[0]] = None
		newCff.setRow(entry[0], entry[1], "boundary")
		newCff.setRow(entry[0], entry[2], "boundary")

	keys = cuttingPoints.keys()
	keys.sort(lambda a,b: cmp(float(a[1]),float(b[1])))
	recKeys = recordingDict.keys()
	lastBoundary = cff.getRows()[-1][1]
	#print keys

	# discarding of the segments
	for row in cff.getRows():
		if row[0] in recKeys:
			if float(row[1]) < float(keys[0][1]) and ( float(keys[0][1]) < float(lastBoundary) ):
				newCff.setRow(row[0],row[1],row[2],row[3])
			elif  ( float(row[1]) >= float(keys[0][1]) ) and ( float(row[1]) <= float(cuttingPoints[keys[0]]) ):
				pass
			elif ( float(row[1]) > float(cuttingPoints[keys[0]]) ):
				if ( float(row[1]) < float(keys[1][1]) ) and ( float(keys[1][1]) < float(lastBoundary) ) :
					newCff.setRow(row[0],row[1],row[2],row[3])
				else:
					del keys[0]
					#newCff.setRow(row[0],row[1],row[2],row[3])
			elif ( float(row[1]) < float(lastBoundary) ) or row[2] == "boundary":
				newCff.setRow(row[0],row[1],row[2],row[3])
			else:
				pass
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])

	return removeEmptySegs(newCff)

#set segments with a confidence score < threshold and a duration > minSegLength to [UNKNOWN]
def mapWordsToUnknownWithConf(cff, threshold=0.5, minSegLength=0.2):
	oldCff = cff
	newCff = Cff()
	rows = oldCff.getRows()
	simpleSeg = SimpleSeg()
	segLength = SimpleSeg()
	pat = re.compile('\[UNKNOWN\]')
	cuttingPoints = {}
	confDict = {}
	mapDict = {}

	for row in rows:
		if row[2] == 'score':
			confDict = row[3]
			currentConfidence = confDict['confidence']
			if float(currentConfidence) < float(threshold):
				mapDict[(row[0], row[1])] = None
			else:
				pass
		else:
			pass

	for row in rows:
		segLength.setKey(row[0], row[1])

		if row[2] == 'boundary' and segLength.orth != None:
			if mapDict.has_key((row[0], segLength.start)):
				mapDict[(row[0], segLength.start)] = float(float(segLength.end) - float(segLength.start))
			segLength.reset()
		if row[2] == 'orth':
			segLength.appendOrth(row[3])

	#blub = mapDict.keys()
	#blub.sort()
	#for i in blub:
	#	if mapDict[i] != None:
	#		print i," ",mapDict[i]

	discard = False
	for row in rows:
		if row[2] == 'boundary' and mapDict.has_key((row[0], row[1])):
			#print row[1]," ",mapDict[(row[0], row[1])]
			if mapDict[(row[0], row[1])] != None:
				if float(mapDict[(row[0], row[1])]) > float(minSegLength):
					discard = True
				else:
					discard = False
			else:
				discard = False

		if row[2] == 'orth' and mapDict.has_key((row[0], row[1])) and discard == True:
			newCff.setRow(row[0],row[1],row[2], '[UNKNOWN]')
		else:
			newCff.setRow(row[0],row[1],row[2],row[3])

	for row in newCff.getRows():
		simpleSeg.setKey(row[0], row[1])

		if row[2] == 'boundary' and simpleSeg.orth != None:
			if re.search(pat, str(simpleSeg.orth.encode('UTF-8'))) != None:
				cuttingPoints[(simpleSeg.recording, simpleSeg.start)] = simpleSeg.end
			simpleSeg.reset()
		if row[2] == 'orth':
			simpleSeg.appendOrth(row[3])

	simpleFilterStats(cuttingPoints, newCff)

	return newCff

#########################################################################
# simple statistics

class Statistics:
	def __init__(self):
		self.recording = None
		self.segmentDurationSum = 0.0
		self.totalSegments = 0
		self.start = None
		self.end = None
		self.reset()

	def reset(self):
		if self.start != None and self.end != None:
			self.duration = float(self.end) - float(self.start)
			self.segmentDurationSum += self.duration
			self.totalSegments += 1
		self.start = None
		self.end = None
		self.orth = None

	def setValues(self, recording, time):
		if self.recording != recording:
			self.recording = recording
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

def getTimeInHours(seconds):
	hours = int(seconds / 3600)
	minutes = int((seconds - 3600 * hours)/ 60 )
	seconds = round(float(seconds - hours * 3600 - minutes * 60),2)
	return(str(str(hours) + 'h ' + str(minutes) + 'm ' + str(seconds) + 's'))

def getPercent(w, g):
	return str(round(float(float(w)/float(g) * 100),2))

def simpleFilterStats(cuttingDict, cff):
	statistics = Statistics()
	for row in cff.getRows():
		statistics.setValues(row[0], row[1])
		if (row[2] == 'boundary' or row[2] == 'boundary_sync') and statistics.orth != None:
			statistics.reset()
		if row[2] == 'orth':
			statistics.appendOrth(row[3])
	totalDuration = 0.0
	segmentCounter = 0
	keys = cuttingDict.keys()
	for key in keys:
		segmentCounter += 1
		totalDuration += float(cuttingDict[key]) - float(key[1])
	print 'Filter Statistics:'
	print 'discarded %i of %i segments with a total duration of %s (corresponds to %s%% of the original %s)' \
		  % (segmentCounter, statistics.totalSegments, getTimeInHours(totalDuration), getPercent(totalDuration, statistics.segmentDurationSum), getTimeInHours(statistics.segmentDurationSum))

def crosswordViolations(cff):
	statistics = Statistics()
	previousOrth = None
	cwViolated = 0
	cwTotal = 0

	for row in cff.getRows():
		statistics.setValues(row[0], row[1])
		if (row[2] == 'boundary' or row[2] == 'boundary_sync'):# and statistics.orth != None:
			statistics.reset()
			if row[2] == 'boundary':
				previousOrth = None
		if row[2] == 'orth':
			if previousOrth == None:
				previousOrth = row[3]
			elif previousOrth != None and (previousOrth == '[IGNORE]' or previousOrth.find('[') == -1) :
				if (row[3] == '[IGNORE]' or row[3].find('[') == -1) :
					cwTotal += 1
					if (previousOrth == '[IGNORE]' and row[3] != '[IGNORE]') or (previousOrth != '[IGNORE]' and row[3] == '[IGNORE]'):
						cwViolated += 1
			previousOrth = row[3]
			statistics.appendOrth(row[3])

	print 'Crossword-Context Violations:'
	print 'violated %i of %i possible crossword-contexts (corresponds to %s%%)' \
		  % (cwViolated, cwTotal, getPercent(cwViolated, cwTotal))


#########################################################################

def main(options, args):

	def printFilterMethods():
		print
		print '\tSegmentation methods:'
		print
		print '\t\tinsert-syncs'
		print '\t\t\tinsert as many sync marks as possible'
		print
		print '\t\tcreate-maximal-segments'
		print '\t\t\tcreate as many segments as possible'
		print
		print '\t\tcreate-minimal-segments'
		print '\t\t\tjoin as many segments as possible'
		print
		print '\t\tadd-boundary-between-silences-longer:[SECONDS]'
		print '\t\t\tadd boundary marker between silences that are longer than [SECONDS]'
		print
		print '\t\tadd-boundary-between-cumulated-nonSpeech-longer:[SECONDS]'
		print '\t\t\tadd boundary in the middle between a series of non-speech tokens longer than [SECONDS]'
		print
		print '\t\tadd-boundary-between-nonSpeech-advanced-longer:[SECONDS]:[MAXSEGLENGTH]'
		print '\t\t\tadd boundary in the middle between a series of non-speech tokens longer than [SECONDS]'
		print '\t\t\tand repeat this process with decreasing [SECONDS] by 0.1s until all segments are'
		print '\t\t\tsmaller than [MAXSEGLENGTH] or [SECONDS] is decreased up to 0.1'
		print
		print
		print '\tFilter methods:'
		print
		print '\t\tremove-orth-longer:[SECONDS]'
		print '\t\t\tremove orthography from segments that are longer then [SECONDS]'
		print
		print '\t\tkeep-orth-by-vocabulary:[WORDLIST]'
		print '\t\t\tremove orthogrphy which contains a word from WORDLIST'
		print
		print '\t\torth-to-phon:[LEXICON]'
		print '\t\t\tconverts orthography to phoneme sequence using LEXICON'
		print
		print '\t\tescape-orth-for-xml'
		print "\t\t\tescapes special xml characters: '<', '>' and '&' in orthography"
		print
		print '\t\tdistribute-orth'
		print '\t\t\tif a segment contains more than one word, the words are distributed linearly among the segment'
		print '\t\t\t(important only for NIST-scoring: .ctm files)'
		print
		print '\t\tremove-segs-confidence:[THRESHOLD]'
		print '\t\t\tremove segments with a confidence score lower than THRESHOLD'
		print
		print '\t\tremove-orth-by-score:[SCORE-NAME]:[LOWER-BOUND]:[UPPER-BOUND]'
		print '\t\t\tremove orth due to the score'
		print
		print '\t\tremove-segs-silence-between-words'
		print '\t\t\tremove segments which contain the Silence Token between two words'
		print
		print '\t\tremove-segs-lm:[LM-FILE]:[NGRAM-ORDER]:[THRESHOLD]:[NGRAM-FILE]{0,1}'
		print '\t\t\tread all nGrams with order NGRAM-ORDER from the ARPA language model LM-FILE into a prefix-tree,'
		print '\t\t\tcalculate segmentwise coverage and discard a segment if its coverage is below THRESHOLD;'
		print '\t\t\tif NGRAM-FILE is given, the prefix-tree will be stored pickled in NGRAM-FILE'
		print
		print '\t\tremove-segs-text:[TEXT-FILE]:[NGRAM-ORDER]:[THRESHOLD]:[NGRAM-FILE]{0,1}'
		print '\t\t\tread TEXT-FILE and store nGrams with order NGRAM-ORDER in a prefix-tree,'
		print '\t\t\tcalculate segmentwise coverage and discard a segment if its coverage is below THRESHOLD;'
		print '\t\t\tif NGRAM-FILE is given, the prefix-tree will be stored pickled in NGRAM-FILE'
		print
		print '\t\tnormalize-conf-over-segs'
		print '\t\t\tif confidence scores are given per word, a segmentwise confidence score is calculated and assigned to every word'
		print
		print '\t\tremove-empty-segs'
		print '\t\t\tempty segments are removed'
		print
		print '\t\tremove-segs-noise'
		print '\t\t\tall segments containing only special lemmas enclosed in square brackets [] are removed'
		print
		print '\t\tremove-segs-timepoints:[FILE]'
		print '\t\t\tif the file has the form RECORDING STARTTIME ENDTIME, all rows within these timespans are removed from'
		print '\t\t\tthe cff file and appropriate boundaries are inserted'
		print
		print '\t\treplace-words-confidence:[THRESHOLD]:[LEXICON-FILE]'
		print '\t\t\tif confidence scores per word are given, words with a confidence lower than THRESHOLD wil be replaced with [IGNORE]'
		print '\t\t\tif [LEXICON] is provided, a word will be replaced by n [IGNORE] tokens, where n is the number of phonemes of the word'
		print
		print '\t\tmap-words-confidence:[THRESHOLD]'
		print '\t\t\tif confidence scores per word are given, words with a confidence lower than THRESHOLD wil be replaced with [UNKNOWN]'
		print
		print '\t\tinsert-crossword-context'
		print '\t\t\ton every position where a posible crossword-context is violated due to an [IGNORE] token,'
		print '\t\t\ta [CONTEXT] token is inserted'
		print
		print '\t\tget-sgml-data:[SGML-FILE]'
		print '\t\t\talignment data from a NIST-sgml file is read into a dictionary'
		print
		print '\t\tsentb2segb:[CONTEXT]'
		print '\t\t\tconvert sentence boundaries to segment boundaries'
		print '\t\t\tif [CONTEXT] (boolean) is True, only those SBs are replaced which can not violate a crossword context'
		print
		print '\t\tremoveSBs'
		print '\t\t\tremoves all sentence boundaries'
		print


	if options.listMethods:
		printFilterMethods()
		sys.exit(1)

	################
	# corpus input
	################

	cff = Cff()
	if options.inputFile == 'stdin':
		cff.readRows(sys.stdin)
	else:
		if options.inputFile.endswith('.cff'):
			cff.readRows(open(options.inputFile, 'rb'))
		else:
			print >> sys.stderr, 'File '+options.inputFile+" does not end with '.cff'"


	########################
	# parse filter options
	########################

	for arg in args:
		################
		# segmentation
		################
		if arg == 'insert-syncs':
			cff = insertSync(cff)

		elif arg == 'create-maximal-segments':
			cff = minimalSegmentation(cff)

		elif arg == 'create-minimal-segments':
			cff = mergeSegments(cff)

		elif arg.startswith('add-boundary-between-silences-longer:'):
			cff = rawSegmentation(cff, float(arg[37:]))

		elif arg.startswith('add-boundary-between-cumulated-nonSpeech-longer:'):
			cff = advancedSegmentation(cff, float(arg[48:]))

		elif arg.startswith('add-boundary-between-nonSpeech-advanced-longer:'):
			argList = arg.split(':')
			nonSpeechLength = argList[1]
			maxSegmentLength = argList[2]
			cff = advancedSegmentationWrapper(cff, float(nonSpeechLength), float(maxSegmentLength))

		###########
		# filters
		###########
		elif arg.startswith('remove-orth-longer:'):
			cff = removeOrthIfLonger(cff, float(arg[19:]))

		elif arg.startswith('remove-orth-shorter:'):
			cff = removeOrthIfShorter(cff, float(arg[20:]))

		elif arg.startswith('keep-orth-by-vocabulary:'):
			knownWordList = []
			file = open(arg[24:] ,'r')
			for line in file.readlines():
				line = line.decode('UTF-8')
				knownWordList.append(line[:-1])
			cff = keepOrthByVocabulary(cff, knownWordList)

		elif arg.startswith('orth-to-phon:'):
			cff = orth2phon(cff, arg[13:])

		elif arg == 'escape-orth-for-xml':
			cff = escapeOrthForXml(cff)

		elif arg == 'distribute-orth':
			cff = distributeOrth(cff)

		elif arg.startswith('remove-segs-confidence:'):
			cff = removeSegsWithConf(cff, float(arg[23:]), options.isInfo)

		elif arg.startswith('remove-orth-by-score:'):
			scoreName = arg.split(':')[1]
			lowerBound = float(arg.split(':')[2])
			upperBound = float(arg.split(':')[3])
			cff = removeOrthByScore(cff, scoreName, lowerBound, upperBound)

		elif arg == 'remove-segs-silence-between-words':
			cff = removeSegsWithSil(cff, options.isInfo)

		### TODO : lm (not sufficently tested) ###
		elif arg.startswith('remove-segs-lm:'):
			argList = arg.split(':')
			lm = argList[1]
			nGramOrder = argList[2]
			threshold = argList[3]
			try:
				pTree = argList[4]
			except:
				pTree = None
			cff = removeSegsWithLm(cff, lm, nGramOrder, threshold, pTree, options.isInfo)

		elif arg.startswith('remove-segs-dict:'):
			cff = removeWithDictFile(cff, str(arg[17:]), options.isInfo)

		elif arg.startswith('remove-segs-timepoints:'):
			cff = removeWithDictFileTimepoints(cff, str(arg[23:]), options.isInfo)

		elif arg.startswith('remove-segs-text:'):
			argList = arg.split(':')
			text = argList[1]
			nGramOrder = argList[2]
			threshold = argList[3]
			try:
				pTree = argList[4]
			except:
				pTree = None
			cff = removeSegsWithTextFile(cff, text, nGramOrder, threshold, pTree, options.isInfo)

		elif arg == 'normalize-conf-over-segs':
			cff = assignNormalizedWPP(cff)

		elif arg == 'remove-empty-segs':
			cff = removeEmptySegs(cff)

		elif arg == 'remove-segs-noise':
			cff = removeNoiseSegments(cff, options.isInfo)

		elif arg.startswith('replace-words-confidence:'):
			argList = arg.split(':')
			threshold = argList[1]
			try:
				lexicon = argList[2]
			except:
				lexicon = None
			cff = replaceWordsConfidence(cff, float(threshold), lexicon, options.isInfo)

		elif arg.startswith('map-words-confidence:'):
			argList = arg.split(':')
			threshold = argList[1]
			minSegLength = argList[2]
			cff = mapWordsToUnknownWithConf(cff, float(threshold), float(minSegLength))

		elif arg == 'insert-crossword-context':
			cff = insertCrosswordContext(cff)

		elif arg.startswith('get-sgml-data:'):
			cff = addSgmlToCff(cff, arg[14:])

		elif arg.startswith('sentb2segb'):
			argList = arg.split(':')
			try:
				context = argList[1]
			except:
				context = None
			cff = sentenceBoundaries2segmentBoundaries(cff, context)

		elif arg == 'removeSBs':
			cff = removeSentenceBoundaries(cff)

		else:
			print >> sys.stderr, "Unknown filter option: "+arg

	#################
	# corpus output
	#################

	if options.outputFile == 'stdout':
		file = sys.stdout
	else:
		file = open(options.outputFile, 'wb')
	cff.writeRows(file)


if __name__ == '__main__':

	optParser = OptionParser(usage="usage: %prog [FILTERMETHOD...]", version="%prog 0.1")
	optParser.set_description("Corpus filter tool for .cff files")
	optParser.add_option("-o", "--output-file", metavar="FILE", action="store", type="string",
						 dest="outputFile", help="Set the output file, default=stdout", default='stdout')
	optParser.add_option("-i", "--input-file", metavar="FILE", action="store", type="string",
						 dest="inputFile", help="Set the input file, default=stdin", default='stdin')
	optParser.add_option("-L", "--list-methods", action="store_true",
						 dest="listMethods", help="Print list of available filter methods", default=False)
	optGroupS = OptionGroup( optParser, 'Simple Filter Statistics' )
	optGroupS.add_option("-s", "--statistics", action="store_true",
						 dest="isInfo", help="activates some statistics about discarded Elements due to an applied filter", default=False)

	optParser.add_option_group(optGroupS)
	options, args = optParser.parse_args()
	main(options, args)
