import os
import random
import re
import string
import sys
from os.path import abspath
from sets import Set
from xml import sax
from xml.sax import saxutils
from xmlWriterLib import openXml, closeXml
from miscLib import zopen, zclose, uopen, uclose, ToAsciiConverter

# ******************************************************************************
# almost like /dev/null
def dummy(*ignore):
    pass

# ***************************** Costa Log File **************************************
class CostaLogParser(sax.handler.ContentHandler):
    def __init__(self, costaLog):
	# handler
	self.costaLog = costaLog
	self.isLexiconLine = False
	self.isMissingWord = False
	self.content = ''
	# parser
	self.parser = sax.make_parser()
	self.parser.setFeature(sax.handler.feature_namespaces, 0)
	self.parser.setContentHandler(self)

    def endDocument(self):
	if  self.costaLog.blissLexiconFile == None:
	    err = 'analysis of costa log file failed: no bliss lexicon found'
	    raise err

    def startElement(self, name, attr):
	self.content = ""
	if self.isMissingWord:
	    pass
	elif self.isLexiconLine:
	    # something went wrong ...
	    pass
	elif name == 'lemma' and attr.get('special', '') == 'unknown':
	    self.isMissingWord = True
	elif name == "information" and attr.get('component', '') == 'costa.lexicon':
	    self.isLexiconLine = True

    def endElement(self, name):
	cdata = self.content.encode('iso-8859-1').strip()
	self.content = ""
	if self.isMissingWord:
	    if name == 'orth':
		self.costaLog.missingWordList.append(cdata)
	    elif name == 'lemma':
		self.isMissingWord = False
	elif self.isLexiconLine:
	    if name == 'information':
		m = re.compile(r'reading lexicon from file "(.*?)"').search(cdata)
		if m:
		    self.costaLog.blissLexiconFile = m.group(1)
		self.isLexiconLine = False

    def characters(self, content):
	self.content += content

    def parse(self, file):
	fd = zopen(file, 'r')
	self.parser.parse(fd)
	zclose(fd)


class CostaLog:
    def __init__(self, file):
	self.blissLexiconFile = None
	self.missingWordList = []
	parser = CostaLogParser(self)
	parser.parse(file)

    def getLexiconFile(self):
	return self.blissLexiconFile

    def getMissingWordList(self):
	return self.missingWordList


# ***************************** Lexica **************************************
class BlissLexiconParser(sax.handler.ContentHandler):
    def __init__(self, addPhoneme, addLemma):
	# handler
	self.addPhoneme = addPhoneme
	self.addLemma   = addLemma
	self.resetPhoneme()
	self.resetLemma()
	self.tokenList = []
	self.isTokenSeq = False
	self.content = ''
	self.startHandler = dummy
	self.endHandler   = self.endPhonemeHandler
	# parser
	self.parser = sax.make_parser()
	self.parser.setFeature(sax.handler.feature_namespaces, 0)
	self.parser.setContentHandler(self)

    def resetPhoneme(self):
	self.symbolList = []
	self.variation = None

    def resetLemma(self):
	self.special = None
	self.orthList = []
	self.phonSeqList = []
	self.phonScoreDict = {}
	self.syntTokenSeq = None
	self.evalTokenSeq = None

    def startElement(self, name, attr):
	self.content = ''
	self.startHandler(name, attr)

    def endElement(self, name):
	cdata = self.content.strip()
	self.content = ''
	self.endHandler(name, cdata)

    def endPhonemeHandler(self, name, cdata):
	if name == 'symbol':
	    self.symbolList.append(cdata)
	elif name == 'variation':
	    self.variation = cdata
	elif name == 'phoneme':
	    self.addPhoneme(self.symbolList, self.variation)
	    self.resetPhoneme()
	elif name == 'phoneme-inventory':
	    self.startHandler = self.startLemmaHandler
	    self.endHandler   = self.endLemmaHandler
	else:
	    print >> sys.stderr, 'WARNING: unknown or unexpected element:', name

    def startLemmaHandler(self, name, attr):
	if self.isTokenSeq:
	    pass
	elif name == 'synt' or name == 'eval':
	    self.isTokenSeq = True
	    self.tokenList = []
	elif name == 'lemma':
	    self.special = attr.get('special', None)
	elif name == 'phon':
	    self.phonScore = float(attr.get('score', 0.0))

    def endLemmaHandler(self, name, cdata):
	if self.isTokenSeq:
	    if name == 'tok':
		self.tokenList.append(cdata)
	    elif  name == 'synt':
		self.syntTokenSeq = tuple(self.tokenList)
		self.isTokenSeq = False
	    elif  name == 'eval':
		self.evalTokenSeq = tuple(self.tokenList)
		self.isTokenSeq = False
	elif name == 'orth':
	    self.orthList.append(cdata)
	elif name == 'phon':
	    phonTranscription=tuple(cdata.split())
	    self.phonSeqList.append(phonTranscription)
	    self.phonScoreDict[phonTranscription]=self.phonScore
	elif name == 'lemma':
	    self.addLemma(self.special, self.orthList, self.phonSeqList, self.syntTokenSeq, self.evalTokenSeq, self.phonScoreDict)
	    self.resetLemma()
	elif name == 'lexicon':
	    self.startHandler = dummy
	    self.endHandler   = dummy
	else:
	    print >> sys.stderr, 'WARNING: unknown or unexpected element:', name

    def characters(self, content):
	self.content += content

    def parse(self, file):
	fd = zopen(file, 'r')
	self.parser.parse(fd)
	zclose(fd)


class PlainLexiconParser:
    def defaultRowFilter(row):
	s = row.split(None, 1)
	return None,                                                       \
	       [ s[0].split('\\', 1)[0] ],                                 \
	       [ tuple(p.split()) for p in s[1].split('\\') if p.strip() ]
    defaultRowFilter = staticmethod(defaultRowFilter)

    def tabSeparatedRowFilter(row):
	s = row.split('\t', 1)
	return None,                                                                                 \
	       [ s[0].strip().split('\\', 1)[0] ],                                                   \
	       [ tuple(p.split()) for p in s[1].lstrip().split('\t', 1)[0].split('\\') if p.strip() ]
    tabSeparatedRowFilter = staticmethod(tabSeparatedRowFilter)

    def tabSimpleRowFilter(row):
	s = row.split('\t', 1)
	return None, \
		[ s[0].strip() ], \
		[ tuple(s[1].strip().split()) ]
    tabSimpleRowFilter = staticmethod(tabSimpleRowFilter)

    def callhomeRowFilter(row):
	s = row.split()
	return None,                                       \
	       [ s[0] ],                                   \
	       [ tuple(p) for p in s[2].split('//') if p ]
    callhomeRowFilter = staticmethod(callhomeRowFilter)

    def orthToGraphemeFilter(row):
	return None, [ row ], [ tuple(row) ]
    orthToGraphemeFilter = staticmethod(orthToGraphemeFilter)

    class PhraseFilter:
	def __init__(self, filter):
	    self.filter = filter

	def __call__(self, row):
	    lemmaData = self.filter(row)
	    orth, phonSeqList = lemmaData[1:3]
	    orth = orth[0].replace('_', ' ').split()
	    if len(orth) > 1:
		return None,                                \
		       [ '_'.join(orth), ' '.join(orth) ],  \
		       phonSeqList,                         \
		       tuple(orth),                         \
		       tuple(orth)
	    else:
		return lemmaData

    def __init__(self, encoding, addLemma, rowFilter, options):
	self.encoding = encoding
	self.addLemma = addLemma
	self.skipHeader = 'header' in options
	if 'phrases' in options:
	    self.parseRow = PlainLexiconParser.PhraseFilter(rowFilter)
	else:
	    self.parseRow = rowFilter

    def parse(self, file):
	fd = uopen(file, self.encoding, 'r')
	fd_it = iter(fd)
	try:
	    if self.skipHeader:
		fd_it.next()
	    for row in fd_it:
		row = row.strip()
		if row:
		    lemmaData = self.parseRow(row)
		    if lemmaData:
			self.addLemma(*lemmaData)
##		    try:
##			if lemmaData:
##			    self.addLemma(*lemmaData)
##		    except:
##			print >> sys.stderr, 'WARNING: could not parse "' + row + '"'
	except StopIteration:
	    pass
	uclose(fd)



class LmNoneTokenGenerator:
    def syntTokenSeq(self, orth, evalTokenSeq = None):
	return None
    def evalTokenSeq(self, orth, syntTokenSeq = None):
	return None

class LmIdTokenGenerator:
    def syntTokenSeq(self, orth, evalTokenSeq = None):
	return tuple( [orth] )
    def evalTokenSeq(self, orth, syntTokenSeq = None):
	return tuple( [orth] )


class BlissPhoneme:
    __slots__ = ('symbols', 'variation', 'features')
    def __init__(self, symbolList, variation, features = []):
	self.symbols = symbolList
	self.variation = variation
	self.features = features

# important:
# phonSeqList, syntTokenSeq, evalTokenSeq must be tuples (not lists)
# in order to persistent hash values
class BlissLemma:
    __slots__ = ('special', 'orthList', 'phonSeqList', 'syntTokenSeq', 'evalTokenSeq')
    def __init__(self, special, orthList, phonSeqList, syntTokenSeq, evalTokenSeq, phonScoreDict):
	self.special = special
	self.orthList = orthList
	self.phonSeqList = phonSeqList
	self.syntTokenSeq = syntTokenSeq
	self.evalTokenSeq = evalTokenSeq
	self.phonScoreDict = phonScoreDict


class BlissLexicon:
    def __init__(self, encoding, generator = LmNoneTokenGenerator()):
	self.encoding = encoding
	self.phonList  = []
	self.phonDict = {}
	self.lemmaList = []
	self.specialDict = {}
	self.orthDict = {}
	self.syntTokenSet = Set()
	self.evalTokenSet = Set()
	self.generator = generator
	# options
	self.conv = None
	self.isDumpVariants = False
	self.isDumpSpecials = False
	self.isWarning = True
	self.isExceptionOnWarning = False
	# out
	self.xmlErr = None

    blissTokenRE = re.compile(r'(?:<.+?>)|(?:\[.+?\])|(?:\S+)')
    def countPhonemes(self, fileIn, fileOut):
	wordN, phonemeN = 0, 0
	fdIn = uopen(fileIn, self.encoding, 'r')
	for s in fdIn:
	    for token in self.blissTokenRE.findall(s):
		if token[0] not in '<[':
		    lemma = self.orthDict.get(token)
		    if lemma is not None:
			if lemma.phonSeqList is not None:
			    if len(lemma.phonSeqList) > 0:
				wordN += 1
				phonemeN += len(lemma.phonSeqList[0])
	uclose(fdIn)
	fdOut = uopen(fileOut, self.encoding, 'w')
	print >> fdOut, '#words:               ', wordN
	print >> fdOut, '#phonme:              ', phonemeN
	print >> fdOut, '#phoneme/word on avg.:', float(phonemeN)/float(wordN)
	uclose(fdOut)



    # ############################### options ######################################
    def setEncoding(encoding):
	self.encoding = encoding

    def setNormal(self):
	self.conv = None

    def setUpper(self):
	self.conv = string.upper

    def setLower(self):
	self.conv = string.lower

    def setDumpVariants(self, isDumpVariants):
	self.isDumpVariants = isDumpVariants

    def setDumpSpecials(self, isDumpSpecials):
	self.isDumpSpecials = isDumpSpecials

    def setWarning(isWarning):
	self.isWarning = isWarning

    def setExceptionOnWarning(isExceptionOnWarning):
	self.isExceptionOnWarning = isExceptionOnWarning


    # ############################### auxiliary ######################################
    def encoding_(self, encoding = None):
	if encoding is None:
	    return self.encoding
	else:
	    return encoding

    def escapeForXml(self, data):
	data = data.replace('&', '&amp;')
	data = data.replace('<', '&lt;')
	data = data.replace('>', '&gt;')
	return data


    # ############################### noises #######################################
    # format of noise file: (\s*((\[[^\]]+\])|({[^}]+})|(\S+))*\s*\n)*, e.g.
    # 0: [speaker laughing] [speaker coughing]
    # 1: eh oh uh
    # 2: {fil}
    # all phonemes in one row are mapped to a single, new phoneme;
    # such phonemes start with a G(=>Geraeusch)
    def getNewNoisePhonemeSymbol(self, p = 'Gnew'):
	while p in self.phonDict:
	    p = 'G' \
		+ chr(random.randint(ord('a'), ord('z'))) \
		+ chr(random.randint(ord('a'), ord('z'))) \
		+ chr(random.randint(ord('a'), ord('z')))
	return p

    def mergeNoiseFile(self, noiseFile, encoding = None, mapToSilence = False):
	noiseRE = re.compile(r'(?:\[[^\]]+\])|(?:{[^}]+})|(?:\S+)')
	fd = uopen(noiseFile, encoding_(encoding), 'r')
	if mapToSilence:
	    ol = []
	    for row in fd:
		ol.extend(noiseRE.findall(row))
	    if ol:
		self.addLemma(None, ol, [ tuple(['si']) ], tuple(), tuple())
	else:
	    for row in fd:
		ol = noiseRE.findall(row)
		if ol:
		    symbol = self.getNewNoisePhonemeSymbol('G' + ol[0].replace(' ', '').replace('[', '').replace('{', '')[:3])
		    self.addPhoneme([symbol], 'none')
		    self.addLemma(None, ol, [ tuple([symbol]) ], tuple(), tuple())
	uclose(fd)

# ############################### format #######################################
    def lowerOrthography(self):
	for l in self.lemmaList:
	    if l.special is None:
		l.orthList = [ o.lower() for o in l.orthList ]
	orthDict = {}
	for o, l in self.orthDict.iteritems():
	    if l.special is None:
		o = o.lower()
	    if o in orthDict:
		for i in range(len(l.phonSeqList)):
		    p = l.phonSeqList[i]
		    if p not in orthDict[o].phonSeqList:
			orthDict[o].phonSeqList.append(p)
			orthDict[o].phonScoreDict[p] = l.phonScoreDict[p]

	    else:
		orthDict[o] = l
	self.orthDict = orthDict

    def upperOrthography(self):
	for l in self.lemmaList:
	    if l.special is None:
		l.orthList = [ o.upper() for o in l.orthList ]
	orthDict = {}
	for o, l in self.orthDict.iteritems():
	    if l.special is None:
		o = o.upper()
	    if o in orthDict:
		err = 'ERROR: upper failed, ' + o + ' became ambigious'
		raise Exception(err)
	    else:
		orthDict[o] = l
	self.orthDict = orthDict

    # ############################### warnings ######################################
    def openXmlErr(self):
	if self.xmlErr is None:
	    self.xmlErr = openXml('stderr', self.encoding)
	    self.xmlErr.open('lextool')

    def closeXmlErr(self):
	if self.xmlErr is not None:
	    self.xmlErr.close('lextool')
	    closeXml(self.xmlErr)

    def warnDuplicatedOrthography(self, orth, lemma1, lemma2):
	self.openXmlErr()
	self.xmlErr.open('warning')
	self.xmlErr.cdata('duplicate orthography: ' + orth)
	self.xmlErr.cdata('orthography exists in lemma')
	self.dumpBlissLemma(self.xmlErr, lemma1)
	self.xmlErr.cdata('and is to be added to lemma')
	if lemma2:
	    self.dumpBlissLemma(self.xmlErr, lemma2)
	    self.xmlErr.close('warning')

    def warnDuplicatedPhonemeSymbol(self, symbol, phon1, phon2 = None):
	self.openXmlErr()
	self.xmlErr.open('warning')
	self.xmlErr.cdata('phoneme symbol conflict: ' + symbol)
	self.xmlErr.cdata('phoneme symbol exists in phoneme')
	self.dumpBlissPhoneme(self.xmlErr, phon1)
	if phon2:
	    self.xmlErr.cdata('and is to be added to phoneme')
	    self.dumpBlissLemma(self.xmlErr, phon2)
	self.xmlErr.close('warning')
	if self.isExceptionOnWarning:
	    err = 'phoneme symbol conflict: ' + symbol
	    raise err
	else:
	    self.xmlErr.open('information')
	    self.xmlErr.cdata('resolve conflict: keep symbol in existing phoneme')
	    self.xmlErr.close('information')


    def warnUnknownPhonemeSymbol(self, symbol):
	self.openXmlErr()
	self.xmlErr.open('warning')
	self.xmlErr.cdata('unknown phoneme symbol: ' + symbol)
	self.xmlErr.close('warning')
	if self.isExceptionOnWarning:
	    err = 'unknown phoneme symbol: ' + symbol
	    raise err
	else:
	    self.xmlErr.open('information')
	    self.xmlErr.cdata('discard pronunciation containing unknown phonemes')
	    self.xmlErr.close('information')


    # ############################### phonems ######################################
    def addPonemeSymbolList_(self, symbolList, phon):
	for s in symbolList:
	    self.phonDict[s] = phon


    # add phonem to phonem-inventory
    def addPhoneme(self, symbolList, variation = None):
	for s in symbolList:
	    if s in self.phonDict:
		self.warnDuplicatedPhonemeSymbol(s, self.phonDict[s])
		break
	else:
	    phon = BlissPhoneme(symbolList, variation)
	    self.phonList.append(phon)
	    self.addPonemeSymbolList_(symbolList, phon)

    # merge phonem into phonem-inventory
    def mergePhoneme(self, symbolList, variation = None):
	if len(symbolList) == 1:
	    self. mergePhoneme_(symbolList, variation)
	else:
	    self. mergePhonemeList_(symbolList, variation)

    # merge single symbol
    def mergePhoneme_(self, symbolList, variation):
	phon = self.phonDict.get(symbolList[0])
	if phon is None:
	    phon = BlissPhoneme(symbolList, variation)
	    self.phonList.append(phon)
	    self.addPonemeSymbolList_(symbolList, phon)
	else:
	    if phon.variation is None:
		phon.variation = variation
	    elif variation is not None and variation != phon.variation:
		raise 'variation conflict'

    # merge multiple symbols
    # -> keep existing symbols in the phoneme they belong to
    # -> build new phoneme out of remaining symbols
    def mergePhonemeList_(self, symbolList, variation):
	_symbolList = []
	for symbol in symbolList:
	    phon = self.phonDict.get(symbol)
	    if phon is None:
		_symbolList.append(symbol)
	    else:
		if phon.variation is None:
		    phon.variation = variation
		elif variation is not None and variation != phon.variation:
		    raise 'variation conflict'
	if _symbolList:
	    phon = BlissPhoneme(_symbolList, variation)
	    self.phonList.append(phon)
	    self.addPonemeSymbolList_(_symbolList, phon)


    # add without any consistency check
    def fastAddPhoneme_(self, symbolList, variation):
	phon = BlissPhoneme(symbolList, variation)
	self.phonList.append(phon)
	self.addPonemeSymbolList_(symbolList, phon)


    def getSilencePhonemeSymbol(self):
	try:
	    silenceLemma = self.specialDict['silence']
	    return silenceLemma.phonSeqList[0][0]
	except:
	    self.openXmlErr()
	    self.xmlErr.open('warning')
	    self.xmlErr.cdata('could not find the silence phoneme; merge default silence lemma')
	    self.xmlErr.close('warning')
	    self.mergeDefaultSilence()
	    silenceLemma = self.specialDict['silence']
	    return silenceLemma.phonSeqList[0][0]


# ############################### lemmata ######################################
    # auxiliary functions
    def checkPhonemes_(self, phonSeqList):
	if phonSeqList is None:
	    return phonSeqList
	checkedPhonSeqList = []
	for phonSeq in phonSeqList:
	    for symbol in phonSeq:
		if symbol not in self.phonDict:
		    self.warnUnknownPhonemeSymbol(symbol)
		    break
	    else:
		checkedPhonSeqList.append(phonSeq)
	return checkedPhonSeqList

    def mergePhonemes_(self, phonSeqList):
	if phonSeqList != None:
	    for phonSeq in phonSeqList:
		for symbol in phonSeq:
		    if symbol not in self.phonDict:
			self.fastAddPhoneme_([ symbol ], None)

    def addOrthographyList_(self, orthList, lemma):
	for orth in orthList:
	    self.orthDict[orth] = lemma

    def addPhonemeSeqList_(self, phonSeqList, lemma):
	pass

    def addSyntAndEvalTokenSeq_(self, syntTokenSeq, evalTokenSeq, lemma):
	if syntTokenSeq is None:
	    syntTokenSeq = lemma.orthList[0].split()
	for tok in syntTokenSeq:
	    self.syntTokenSet.add(tok)
	if evalTokenSeq is None:
	    evalTokenSeq = syntTokenSeq
	for tok in evalTokenSeq:
	    self.evalTokenSet.add(tok)

    def convert_(self, conv, orthList, syntTokenSeq, evalTokenSeq):
	orthList = [ conv(o) for o in orthList ]
	if syntTokenSeq is not None:
	    syntTokenSeq = tuple( [ conv(t) for t in syntTokenSeq ] )
	if evalTokenSeq is not None:
	    evalTokenSeq = tuple( [ conv(t) for t in evalTokenSeq ] )
	return orthList, syntTokenSeq, evalTokenSeq

    def addLemma_(self, special, orthList, phonSeqList, syntTokenSeq = None, evalTokenSeq = None, phonScoreDict = None):
	assert orthList
	if self.conv is not None and special is None:
	    orthList, syntTokenSeq, evalTokenSeq = self.convert_(self.conv, orthList, syntTokenSeq, evalTokenSeq)
	if syntTokenSeq == None:
	    syntTokenSeq = self.generator.syntTokenSeq(orthList[0], evalTokenSeq)
	if evalTokenSeq == None:
	    evalTokenSeq = self.generator.evalTokenSeq(orthList[0], syntTokenSeq)
	_orthList = orthList
	isNew = False
	# for-else loop ! (ugly)
	for orth in _orthList:
	    lemma = self.orthDict.get(orth, None)
	    if lemma:
		if special == lemma.special and syntTokenSeq == lemma.syntTokenSeq and evalTokenSeq == lemma.evalTokenSeq:
		    orthList = []
		    for orth in _orthList:
			if orth not in lemma.orthList:
			    lemma.orthList.append(orth)
		    if phonSeqList:
			_phonSeqList = phonSeqList
			phonSeqList = []
			for phonSeq in _phonSeqList:
			    if phonSeq not in lemma.phonSeqList:
				lemma.phonSeqList.append(phonSeq)
				phonSeqList.append(phonSeq)
				if phonScoreDict is not None:
				    lemma.phonScoreDict[phonSeq]=phonScoreDict[phonSeq]  #init score
			    else:
				if phonScoreDict is not None:
				    lemma.phonScoreDict[phonSeq]+=phonScoreDict[phonSeq] #update score
		    syntTokenSeq = ()
		    evalTokenSeq = ()
		    break
	else:
	    lemma = BlissLemma(special, [], phonSeqList, syntTokenSeq, evalTokenSeq, phonScoreDict)
	    isNew = True

	_orthList = []
	for orth in orthList:
	    if orth in self.orthDict:
		self.warnDuplicatedOrthography(orth, self.orthDict[orth], lemma)
	    _orthList.append(orth)
	if _orthList:
	    if isNew:
		self.lemmaList.append(lemma)
		if special is not None:
		    self.specialDict[special] = lemma
	    lemma.orthList.extend(_orthList)
	    self.addOrthographyList_(_orthList, lemma)
	    self.addPhonemeSeqList_(phonSeqList, lemma)
	    self.addSyntAndEvalTokenSeq_(syntTokenSeq, evalTokenSeq, lemma)
	return lemma

    # add without any consistency check
    def fastAddLemma_(self, special, orthList, phonSeqList, syntTokenSeq, evalTokenSeq, phonScoreDict = None):
	if self.conv is not None and special is None:
	    orthList, syntTokenSeq, evalTokenSeq = self.convert_(self.conv, orthList, syntTokenSeq, evalTokenSeq)
	lemma = BlissLemma(special, orthList, phonSeqList, syntTokenSeq, evalTokenSeq, phonScoreDict)
	# insert new lemma without any consistency check
	self.lemmaList.append(lemma)
	if special is not None:
	    self.specialDict[special] = lemma
	#self.addOrthographies_(orthList, lemma)
	for orth in orthList:
	    self.orthDict[orth] = lemma
	#self.addPhonemeSeqList_(phonSeqList, lemma)
	self.addSyntAndEvalTokenSeq_(syntTokenSeq, evalTokenSeq, lemma)
	return lemma

    # merge lemma --> add all missing phonemes, add lemma
    def mergeLemma(self, special, orthList, phonSeqList, syntTokenSeq = None, evalTokenSeq = None, phonScoreDict = None):
	self.mergePhonemes_(phonSeqList)
	return self.addLemma_(special, orthList, phonSeqList, syntTokenSeq, evalTokenSeq, phonScoreDict)

    # add lemma, iff all phonemes are known
    def addLemma(self, special, orthList, phonSeqList, syntTokenSeq = None, evalTokenSeq = None, phonScoreDict = None):
	phonSeqList = self.checkPhonemes_(phonSeqList)
	return self.addLemma_(special, orthList, phonSeqList, syntTokenSeq, evalTokenSeq, phonScoreDict)

    def addCompoundOrthography(self, orthList) :
	orth1 = '_'.join(orthList)
	orth2 = ' '.join(orthList)
	# check if compound does already exist
	lemma = self.orthDict.get(orth1)
	if lemma is not None:
	    if not orth2 in self.orthDict:
		lemma.orthList.append(orth2)
		self.addOrthographyList_([ orth2 ], lemma)
	else:
	    lemma = self.orthDict.get(orth2)
	    if lemma is not None:
		lemma.orthList.append(orth1)
		self.addOrthographyList_([ orth1 ], lemma)
	if lemma is not None:
	    #TODO update synTokenSet and evalTokenSet
##            if lemma.syntTokenSeq is None:
##                lemma.syntTokenSeq = tuple(orthList)
##                lemma.syntEvalSeq = tuple(orthList)
	    return

	# create new compound lemma
	phonSeqList = []
	syntTokenList = []
	evalTokenList = []
	si = self.getSilencePhonemeSymbol()
	for orth in orthList:
	    lemma = self.orthDict.get(orth)
	    if lemma is None:
		self.openXmlErr()
		self.xmlErr.open('error')
		self.xmlErr.cdata('cannot add compound orthography "' + orth1 + '", because "' + orth + '" is unknown')
		self.xmlErr.close('error')
		return
	    if phonSeqList:
		if lemma.phonSeqList:
		    _phonSeqList = []
		    for phonSeq1 in phonSeqList:
			phonSeq1si = phonSeq1 + tuple([si])
			for phonSeq2 in lemma.phonSeqList:
			    _phonSeqList.append(phonSeq1 + phonSeq2)
			    _phonSeqList.append(phonSeq1si + phonSeq2)
		    phonSeqList = _phonSeqList
	    else:
		for phonSeq in lemma.phonSeqList:
		    phonSeqList.append(phonSeq)
	    if lemma.syntTokenSeq is None:
		syntTokenList.append(lemma.orthList[0])
	    else:
		syntTokenList.extend(lemma.syntTokenSeq)
	    if lemma.evalTokenSeq is None:
		evalTokenList.append(lemma.orthList[0])
	    else:
		evalTokenList.extend(lemma.evalTokenSeq)
	self.fastAddLemma_(None, [orth1, orth2], phonSeqList, tuple(['_'.join(syntTokenList)]), tuple(evalTokenList))



    # defaults for special lemmata,
    # i.e. si as silence phoneme,
    #      <UNK> as symbol for unknown words,
    #      <s> and </s> as sentence boundary symbols
    def mergeDefaultLemmas(self):
	self.mergeDefaultSilence()
	self.mergeDefaultUnknown()
	self.mergeDefaultUnintelligible()
	self.mergeDefaultSentenceBoundary()

    def mergeDefaultSilence(self):
	self.mergePhoneme([ 'si' ], 'none')
	self.addLemma('silence', ['', '[SILENCE]'], [ tuple(['si']) ], tuple(), tuple())

    def mergeDefaultUnknown(self):
	self.addLemma('unknown', ['[UNKNOWN]'], None, tuple(['<UNK>']), tuple())

    def mergeDefaultUnintelligible(self):
	self.addLemma(None, ['[???]'], None, tuple(), tuple())

    def mergeDefaultSentenceBoundary(self):
	self.addLemma('sentence-begin', ['[SENTENCE-BEGIN]'], None, tuple(['<s>']) , tuple())
	self.addLemma('sentence-end',   ['[SENTENCE-END]']  , None, tuple(['</s>']), tuple())

    def mergeDefaultPunctationMarks(self):
	self.addLemma(None, ['.', '?', '!']  , None, ('<s>', '</s>'), tuple())


    # ############################### filter ######################################
    # keep lemma if f(<orthography>) is true (or lemma is special)
    def filterOrth_(self, f, special=False):
	lemmaList = self.lemmaList
	self.lemmaList = []
	self.orthDict = {}
	self.syntTokenSet = Set()
	self.evalTokenSet = Set()
	for lemma in lemmaList:
	    if (((not special) and lemma.special is not None) or (special and lemma.special is None)):
		orthList = lemma.orthList
	    else:
		orthList = [ orth for orth in lemma.orthList if f(orth)]
	    if orthList:
		lemma.orthList = orthList
		self.lemmaList.append(lemma)
		self.addOrthographyList_(lemma.orthList, lemma)
		self.addPhonemeSeqList_(lemma.phonSeqList, lemma)
		self.addSyntAndEvalTokenSeq_(lemma.syntTokenSeq, lemma.evalTokenSeq, lemma)

    # keep lemma if f(<syntactic token sequence>) is true (or lemma is special)
    def filterSyntTokenSeq_(self, f):
	lemmaList = self.lemmaList
	self.lemmaList = []
	self.orthDict = {}
	self.syntTokenSet = Set()
	self.evalTokenSet = Set()
	add = False
	for lemma in lemmaList:
	    if lemma.special is not None:
		add = True
	    elif lemma.syntTokenSeq is not None:
		if lemma.syntTokenSeq:
		    add = f(lemma.syntTokenSeq)
		else:
		    add = True
	    else:
		add = f( (lemma.orthList[0],) )
	    if add:
		self.lemmaList.append(lemma)
		self.addOrthographyList_(lemma.orthList, lemma)
		self.addPhonemeSeqList_(lemma.phonSeqList, lemma)
		self.addSyntAndEvalTokenSeq_(lemma.syntTokenSeq, lemma.evalTokenSeq, lemma)

    def intersectOrthFromListFile(self, path, encoding = None):
	fd = uopen(path, self.encoding_(encoding), 'r')
	orthSet = Set([ orth.strip() for orth in fd if orth.strip() ])
	uclose(fd)
	f_intersect = lambda x: x in orthSet
	self.filterOrth_(f_intersect)

    def intersectSyntFromListFile(self, path, encoding = None):
	fd = uopen(path, self.encoding_(encoding), 'r')
	syntTokenSeqSet = Set([ tuple( synt.strip().split() ) for synt in fd if synt.strip() ])
	uclose(fd)
	f_intersect = lambda x: x in syntTokenSeqSet
	self.filterSyntTokenSeq_(f_intersect)

    def removeOrthFromListFile(self, path, encoding = None, special=False):
	fd = uopen(path, self.encoding_(encoding), 'r')
	orthSet = Set([ orth.strip() for orth in fd if orth.strip() ])
	uclose(fd)
	f_intersect = lambda x: x not in orthSet
	self.filterOrth_(f_intersect, special=special)

    def removeSyntFromListFile(self, path, encoding = None):
	fd = uopen(path, self.encoding_(encoding), 'r')
	syntTokenSeqSet = Set([ tuple( synt.strip().split() ) for synt in fd if synt.strip() ])
	uclose(fd)
	f_intersect = lambda x: x not in syntTokenSeqSet
	self.filterSyntTokenSeq_(f_intersect)



    # ############################### lexica ######################################
    # auxiliary functions
    def isBlissLexiconFile_(self, path):
	fd = uopen(path, self.encoding, 'r')
	fdIt, row = iter(fd), ''
	try:
	    while not row:
		row = fdIt.next().strip()
	except StopIteration:
	    pass
	uclose(fd)
	if row.startswith('<?xml'):
	    if path.endswith('lexicon') or path.endswith('lexicon.gz'):
		return True
	    else:
		print >> sys.stderr, 'asssume', path, 'to be in bliss lexicon format'
		return True
	else:
	    return False

    # load (no consistency checks, use merge to load and check a lexicon)
    def fastMergeBlissLexicon(self, file):
	blissLexLoad = BlissLexiconParser(self.fastAddPhoneme_, self.fastAddLemma_)
	blissLexLoad.parse(file)

    def fastMergeLexicon(self, file, options = ''):
	if self.isBlissLexiconFile_(file):
	    self.fastMergeBlissLexicon(file)
	else:
	    self.mergePlainLexicon(file,  PlainLexiconParser.defaultRowFilter, options)

    # add   --> add lemmas, iff for a lemma all phonemes are known
    # merge --> add all missing phonemes, add all lemmas
    def addBlissLexicon(self, file):
	blissLexAdd = BlissLexiconParser(dummy, self.addLemma)
	blissLexAdd.parse(file)

    def mergeBlissLexicon(self, file):
	blissLexMerge = BlissLexiconParser(self.mergePhoneme, self.addLemma)
	blissLexMerge.parse(file)

    def addLexicon(self, file, options = ''):
	if self.isBlissLexiconFile_(file):
	    self.addBlissLexicon(file)
	else:
	    self.addPlainLexicon(file, PlainLexiconParser.defaultRowFilter, options)

    def addPlainLexicon(self, file, rowFilter, options):
	plainLexAdd = PlainLexiconParser(self.encoding, self.addLemma, rowFilter, options)
	plainLexAdd.parse(file)

    def mergePlainLexicon(self, file, rowFilter, options):
	plainLexMerge = PlainLexiconParser(self.encoding, self.mergeLemma, rowFilter, options)
	plainLexMerge.parse(file)

    def mergeLexicon(self, file, options = ''):
	if self.isBlissLexiconFile_(file):
	    self.mergeBlissLexicon(file)
	else:
	    self.mergePlainLexicon(file, PlainLexiconParser.defaultRowFilter, options)

    def addCompoundOrthographies(self, filename):
	fd = uopen(filename, self.encoding, 'r')
	for row in fd:
	    row = row.strip()
	    if row and not row.startswith('#'):
		self.addCompoundOrthography(row.replace('_', ' ').split())
	uclose(fd)


    # ############################### dump ######################################
    def dumpBlissPhoneme(self, xml, phon):
	xml.open('phoneme')
	for symbol in phon.symbols:
	    xml.element('symbol', self.escapeForXml(symbol))
	if phon.variation is not None:
	    xml.element('variation', phon.variation)
	xml.close('phoneme')

    def dumpBlissLemma(self, xml, lemma):
	if lemma.special is not None:
	    xml.open('lemma', special=lemma.special)
	else:
	    xml.open('lemma')
	# dump orthographies
	for orth in lemma.orthList:
	    if orth == '':
		xml.empty('orth')
	    else:
		xml.element('orth', self.escapeForXml(orth))
	# dump phonemes
	if lemma.phonSeqList is not None:
	    for phonSeq in lemma.phonSeqList:
		if len(phonSeq) == 0:
		    xml.empty('phon')
		else:
		    attributes={}
		    if lemma.phonScoreDict is not None:
			attributes['score']=lemma.phonScoreDict[phonSeq]
		    xml.open('phon', attributes)
		    xml.cdata(self.escapeForXml(' '.join(phonSeq)))
		    xml.close('phon')
	# dump synt token
	if lemma.orthList:
	    defaultTokenSeq = [ lemma.orthList[0] ]
	else:
	    defaultTokenSeq = None
	if lemma.syntTokenSeq is not None and lemma.syntTokenSeq != defaultTokenSeq:
	    if len(lemma.syntTokenSeq) == 0:
		xml.empty('synt')
	    else:
		xml.open('synt')
		for tok in lemma.syntTokenSeq:
		    xml.element('tok', self.escapeForXml(tok))
		xml.close('synt')
	# dump eval token
	if lemma.evalTokenSeq is not None and lemma.evalTokenSeq != defaultTokenSeq:
	    if len(lemma.evalTokenSeq) == 0:
		xml.empty('eval')
	    else:
		xml.open('eval')
		for tok in lemma.evalTokenSeq:
		    xml.element('tok', self.escapeForXml(tok))
		xml.close('eval')
	xml.close('lemma')

    def dumpBlissLexicon(self, file):
	phonList = self.phonDict.keys()
	phonList.sort()
	orthList = self.orthDict.keys()
	orthList.sort()

	xml = openXml(file, self.encoding)
	xml.openComment()
	xml.cdata('generated by ' + abspath(sys.argv[0]))
	# xml.cdata(revision)
	xml.closeComment()
	xml.open('lexicon')
	# dump phoneme-inventory
	xml.cdata('')
	xml.openComment()
	xml.cdata('phoneme inventory')
	xml.closeComment()
	xml.open('phoneme-inventory')
	for phon in self.phonList:
	    self.dumpBlissPhoneme(xml, phon)
	xml.close('phoneme-inventory')
	# sort lemmas by first orthographic form
	lemmaKeyList = []
	specialLemmaKeyList = []
	for lemma in self.lemmaList:
	    if lemma.special == None:
		lemmaKeyList.append(lemma.orthList[0])
	    else:
		specialLemmaKeyList.append(lemma.orthList[0])
	lemmaKeyList.sort()
	specialLemmaKeyList.sort()
	# dump lemmas, first special lemmas, then regular lemmas
	xml.cdata('')
	xml.openComment()
	xml.cdata('special lemmas')
	xml.closeComment()
	for key in specialLemmaKeyList:
	    self.dumpBlissLemma(xml, self.orthDict[key])
	xml.cdata('')
	xml.openComment()
	xml.cdata('regular lemmas')
	xml.closeComment()
	for key in sorted(set(lemmaKeyList)):
	    self.dumpBlissLemma(xml, self.orthDict[key])
	xml.close('lexicon')
	closeXml(xml)


    def dumpPlainRow(self, fd, orth, phonSeqList):
	if (phonSeqList != None):
	    if len(phonSeqList) > 0:
		line = orth + "\t" + ' '.join(phonSeqList[0])
		if self.isDumpVariants:
		    for phonSeq in phonSeqList[1:]:
			line += '\\' + ' '.join(phonSeq)
		fd.write(line + '\n')


    def dumpPlainLexicon(self, file):
	orthList = self.orthDict.keys()
	orthList.sort()
	fd = uopen(file, self.encoding, 'w')
	# dump lemmas, first regular lemmas, then special lemmas
	specialOrthList = []
	for key in orthList:
	    lemma = self.orthDict[key]
	    if lemma.special == None:
		self.dumpPlainRow(fd, key, lemma.phonSeqList)
	    else:
		specialOrthList.append(key)
	if self.isDumpSpecials:
	    for key in specialOrthList:
		lemma = self.orthDict[key]
		self.dumpPlainRow(fd, key, lemma.phonSeqList)
	uclose(fd)

    def dumpPronDict(self, file):
	if not self.isDumpVariants:
		print >> sys.stderr, "Warning: variants will be dumped"
	orthList = self.orthDict.keys()
	fd = uopen(file, self.encoding, 'w')
	for orth in orthList:
		phonSeqList = self.orthDict[orth].phonSeqList
		if self.isDumpSpecials or self.orthDict[orth].special == None:
			if len(phonSeqList) > 0:
				for phonSeq in phonSeqList:
					fd.write(orth + '\t' + ' '.join(phonSeq)  +'\n')
			else:
				fd.write(orth + '\t' +'\n')
	uclose(fd)

    def dumpPhonSet(self, file):
	phonList = self.phonDict.keys()
	phonList.sort()
	fd = uopen(file, self.encoding, 'w')
	for phon in self.phonList:
	    print >> fd, ' '.join(phon.symbols)
	uclose(fd)

    def dumpOrthSet(self, file):
	orthList = self.orthDict.keys()
	orthList.sort()
	fd = uopen(file, self.encoding, 'w')
	for orth in orthList:
	    print >> fd, orth
	uclose(fd)

    def dumpSyntTokenSet(self, file):
	syntTokenList = [t for t in self.syntTokenSet]
	syntTokenList.sort()
	fd = uopen(file, self.encoding, 'w')
	for tok in syntTokenList:
	    print >> fd, tok
	uclose(fd)

    def dumpEvalTokenSet(self, file):
	evalTokenList = [t for t in self.evalTokenSet]
	evalTokenList.sort()
	fd = uopen(file, self.encoding, 'w')
	for tok in evalTokenList:
	    print >> fd, tok
	uclose(fd)


    def dumpStatistic(self, file):
	lemmaN                 = 0
	pronunciationVariantsN = 0
	specialLemmaN          = 0
	specialPronunciationVariantsN = 0
	maxOrthListLength      = 0
	maxOrthListLemma       = None
	maxPhonSeqListLength   = 0
	maxPhonSeqListLemma    = None
	maxOrthLemma           = None
	maxOrthLength          = 0
	maxPhonSeqLemma        = None
	maxPhonSeqLength       = 0
	maxSyntTokenSeqLemma   = None
	maxSyntTokenSeqLength  = 0
	maxEvalTokenSeqLemma   = None
	maxEvalTokenSeqLength  = 0
	for lemma in self.lemmaList:
	    lemmaN += 1
	    if lemma.special:
		specialLemmaN += 1
	    if len(lemma.orthList) > maxOrthListLength:
		maxOrthListLemma = lemma
		maxOrthListLength = len(lemma.orthList)
	    if lemma.phonSeqList:
		phonSeqListN = len(lemma.phonSeqList)
		pronunciationVariantsN += phonSeqListN
		if lemma.special:
		    specialPronunciationVariantsN += phonSeqListN
		if phonSeqListN > maxPhonSeqListLength:
		    maxPhonSeqListLemma = lemma
		    maxPhonSeqListLength = len(lemma.phonSeqList)
	    for orth in lemma.orthList:
		tmpLength = len(orth)
		if tmpLength > maxOrthLength:
		    maxOrthLemma = lemma
		    maxOrthLength = tmpLength
	    if lemma.phonSeqList:
		for phonSeq in lemma.phonSeqList:
		    tmpLength = len(phonSeq)
		    if tmpLength > maxPhonSeqLength:
			maxPhonSeqLemma = lemma
			maxPhonSeqLength = tmpLength
	    if lemma.syntTokenSeq:
		tmpLength = len(lemma.syntTokenSeq)
		if tmpLength > maxSyntTokenSeqLength:
		    maxSyntTokenSeqLemma = lemma
		    maxSyntTokenSeqLength = tmpLength
	    if lemma.evalTokenSeq:
		tmpLength = len(lemma.evalTokenSeq)
		if tmpLength > maxEvalTokenSeqLength:
		    maxEvalTokenSeqLemma = lemma
		    maxEvalTokenSeqLength = tmpLength

	xml = openXml(file, self.encoding)
	xml.open('statistic')
	xml.element('lemmas', str(lemmaN))
	xml.element('pronunciation-variants', str(pronunciationVariantsN))
	xml.element('special-lemmas', str(specialLemmaN))
	xml.element('special-pronunciation-variants', str(specialPronunciationVariantsN))
	xml.element('phonemes', str(len(self.phonDict)))
	xml.element('orthographic-forms', str(len(self.orthDict)))
	xml.element('syntactic-token', str(len(self.syntTokenSet)))
	xml.element('evaluation-token', str(len(self.evalTokenSet)))
	if maxOrthListLemma:
	    xml.open('max-orthographic-variants', number=maxOrthListLength)
	    self.dumpBlissLemma(xml, maxOrthListLemma)
	    xml.close('max-orthographic-variants')
	if maxPhonSeqListLemma:
	    xml.open('max-pronunciation-variants', number=maxPhonSeqListLength)
	    self.dumpBlissLemma(xml, maxPhonSeqListLemma)
	    xml.close('max-pronunciation-variants')
	if maxOrthLemma:
	    xml.open('longest-grapheme-sequence', length=maxOrthLength)
	    self.dumpBlissLemma(xml, maxOrthLemma)
	    xml.close('longest-grapheme-sequence')
	if maxPhonSeqLemma:
	    xml.open('longest-phoneme-sequence', length=maxPhonSeqLength)
	    self.dumpBlissLemma(xml, maxPhonSeqLemma)
	    xml.close('longest-phoneme-sequence')
	if maxSyntTokenSeqLemma:
	    xml.open('longest-syntactic-token-sequence', length=maxSyntTokenSeqLength)
	    self.dumpBlissLemma(xml, maxSyntTokenSeqLemma)
	    xml.close('longest-syntactic-token-sequence')
	if maxEvalTokenSeqLemma:
	    xml.open('longest-evaluation-token-sequence', length=maxEvalTokenSeqLength)
	    self.dumpBlissLemma(xml, maxEvalTokenSeqLemma)
	    xml.close('longest-evaluation-token-sequence')
	xml.close('statistic')
	closeXml(xml)


    def sortAndWrite(self, list, file):
	list.sort()
	fd = uopen(file, self.encoding, 'w')
	for row in list:
	    fd.write(row)
	uclose(fd)

    def dumpOldLexicon(self, phonFile, lexFile, orthFile, specialFile):
	phonList = []
	lexList  = []
	orthList = []
	silList  = []
	conv = ToAsciiConverter()
	si = self.getSilencePhoneme()
	for phon in self.phonDict.keys():
	    if not phon == si:
		if len(phon) > 4:
		    print >> sys.stderr, 'WARNING:', phon, 'is to long; phoneme length is restricted to 4(!!!) chars, hail to the old standard system'
		    phon = phon[:4]
		    print >> sys.stderr, '        ', 'try to use', phon, 'instead'
		    if phon in self.phonDict:
			err = 'ERROR: ' + phon + ' does already exist'
			raise err
		phonList.append(phon + ' >> ' + phon + '\n')
	isMul = False
	for key in self.orthDict.keys():
	    orth = key
	    if ' ' in orth:
		print >> sys.stderr, 'WARNING: "' + orth + '" contains space(s); replace them by underscores'
		orth = orth.replace(' ', '_')
	    if '\\' in orth:
		err = 'ERROR:', orth, 'must not contain the \\-character'
		raise err
	    orth = conv.encode(orth)
	    lemma = self.orthDict[key]
	    if lemma.special == None:
		if lemma.phonSeqList == None:
		    isMapToSi = True
		elif len(lemma.phonSeqList) == 0:
		    isMapToSi = True
		elif len(lemma.phonSeqList) == 1 \
		     and len(lemma.phonSeqList[0]) == 1 \
		     and lemma.phonSeqList[0][0] == si:
		    isMapToSi = True
		else:
		    isMapToSi = False
		if isMapToSi:
		    if not isMul:
			phonList.append('mul >> mul\n')
			isMul = True
		    lexList.append(orth + ' mul\n')
		    silList.append(orth + ' -SIL -NOEVAL -NOLEX\n')
		else:
		    for i, phonSeq in enumerate(lemma.phonSeqList):
			if i > 0:
			    ext = '\\' + str(i)
			else:
			    ext = ''
			phonStr = ' '.join(phonSeq).replace(si, 'mul')
			lexList.append(orth + ext + ' ' + phonStr + '\n')
		orthList.append(orth + '\n')
	if len(lexList) > 2**15:
	     print >> sys.stderr, 'WARNING: the lexicon contains (' + str(len(lexList)) + ') entries;'
	     print >> sys.stderr, '         ensure, that the old standard system is compiled for a lexicon of that size'
	self.sortAndWrite(phonList, phonFile)
	self.sortAndWrite(lexList , lexFile)
	self.sortAndWrite(orthList, orthFile)
	self.sortAndWrite(silList , specialFile)

    def dumpCartQuestions(self, file):
	phonList = self.phonDict.keys()
	phonList.sort()
	fd = uopen(file, self.encoding, 'w')
	fd.write('BOUNDARY #\n')
	fd.write('ALL ' + ' '.join(phonList) + '\n\n')
	for symbol in phonList:
	    fd.write(symbol + ' ' + symbol + '\n')
	uclose(fd)

    def createPhrases(self, phrases):
	for phrase in phrases:
	    phraseGenerationScuceeded = True
	    phon = [ tuple() ]
	    for word in phrase.split('_'):
		if self.orthDict.has_key(word):
		    nphon = []
		    for np in self.orthDict[word].phonSeqList:
			nphon += [ (pp + np) for pp in phon]
			for pp in phon:
			    if pp != tuple():
				nphon += [ (pp + (self.getSilencePhonemeSymbol(),) + np) ]
		    phon = nphon
		else:
		    phraseGenerationScuceeded = False
	    if phraseGenerationScuceeded == True:
		self.addLemma(None, [ phrase.replace('_',' ') ], phon, [ phrase ], phrase.split('_'))
	    else:
		print >> sys.stderr, 'ERROR: phrase"' + phrase + '" could not be generated'

# ***************************** convenient functions **************************************
def loadCostaLog(file):
    return CostaLog(file)

# new, empty lexicon
def emptyLexicon(encoding):
    return BlissLexicon(encoding, LmNoneTokenGenerator())
