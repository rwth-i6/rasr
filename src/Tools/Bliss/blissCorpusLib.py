# -*- coding: iso-8859-1 -*-

import os
import re
import sys
from sets import Set as set
from xml import sax
from xml.sax import saxutils
from xmlWriterLib import openXml, closeXml, XmlWriter
from blissLexiconLib import LmIdTokenGenerator
from miscLib import uopen, uclose, zopen, zclose, ToAsciiConverter



class BlissOrthographyConverter:
    """
    Base class to convert raw orthography to bliss orth and vice versa
    """
    def __init__(self):
	self.corpusParser = None

    def squeeze(self, s):
	"all whitspaces to blanks, only one blank in a row"
	return ' '.join(s.strip().split())


    def normalize(self, s):
	"all whitspaces to blanks, only one blank in a row, preceding and succeding blank"
	return " " + self.squeeze(s) + " "


    def transform(self, s, l):
	"""
	applies the transformation contained in the list coded either
	as tuple of two strings (=> replacemanet) or a regular expression
	and a string (=> substitution)
	"""
	for e, r in l:
	    if hasattr(e, "sub"):
		s = e.sub(r, s)
	    else:
		s = s.replace(e, r)
	return s


    blissTokenRE = re.compile(r'(?:<.+?>)|(?:\[.+?\])|(?:\S+)')
    def split(self, s):
	"""
	splits a bliss-xml sentence
	example: '<tag attr=".."> [some noise] text </tag>' -->  ['<tag attr="..">', '[some noise]', 'text', '</tag>']
	"""
	return self.blissTokenRE.findall(s)

    def toUpper(self, s):
	"""
	all words to upper case, but no xml-tags or noise tags (i.e. [noise description])
	"""
	l = []
	for t in self.split(s):
	    if t[0] in '<[':
		l.append(t)
	    else:
		l.append(t.upper())
	return ' '.join(l)


    def toLower(self, s):
	"""
	all words to lower case, but no xml-tags or noise tags(i.e. [noise description])
	"""
	l = []
	for t in self.split(s):
	    if t[0] in '<[':
		l.append(t)
	    else:
		l.append(t.lower())
	return ' '.join(l)


    # merge adjacent elements of same type
    # example: <spelled> A </spelled>  <spelled> B </spelled> -->  <spelled> A B </spelled>
    joinableTags = [
	'noise', 'language', 'hesitation', 'spelled', 'name'
	]
    def setJoinableTagList(self, joinableTags):
	self.joinableTags = joinableTags
    def getJoinableTagList(self):
	return self.joinableTags

    def optimize(self, s):
	lastJoinableOpenTag = ['' for x in self.joinableTags]
	l = self.split(s)
	for i in range(len(l) - 1):
	    token     = l[i]
	    nextToken = l[i+1]
	    if token.startswith('</'):
		for j, tag in enumerate(self.joinableTags):
		    if token == '</' + tag + '>':
			if nextToken == lastJoinableOpenTag[j]:
			    l[i]   = ''
			    l[i+1] = ''
			else:
			    lastJoinableOpenTag[j] = ''
			    break
	    elif token.startswith('<'):
		for j, tag in enumerate(self.joinableTags):
		    if token[1:].startswith(tag):
			lastJoinableOpenTag[j] = token
	return ' '.join(filter(lambda t: (len(t) > 0), l))

    # warning mechanism for unresolved tags (list includes some common transcribtion-tags)
    warnList = [
	re.compile(r'[~!@#$%^&*()+{};:,.]|(?: -)|(?: _)|(?: =)|(?:- )|(?:_ )|(?:= )|(?:\? )|(?:&gt;)|(?:&lt;)')
	]
    def setWarnList(self, warnList):
	self.warnList = warnList
    def getWarnList(self):
	return self.warnList
    def clearWarnList(self):
	self.warnList = []
    def addToWarnList(self, regExpr):
	self.warnList.append(regExpr)

    def warn(self, s, o = 'n/a'):
	for e in self.warnList:
	    m = e.search(s)
	    if m != None:
		print >> sys.stderr, "WARNING:", "(probably) unresolved tag \"" + m.group(0) + "\" detected in"
		print >> sys.stderr, "------------------------------------------------------------------------"
		print >> sys.stderr, "original: ", [ o ]
		print >> sys.stderr, "------------------------------------------------------------------------"
		print >> sys.stderr, "converted:", [ s ]
		print >> sys.stderr, "------------------------------------------------------------------------"
		#print >> sys.stderr


    # dummy
    def getOrthography(self, s):
	# self.warn(s,s)
	return s


class TrlConverter(BlissOrthographyConverter):
    """
    bliss orthography --> trl orthography
    """

    # ########## sentence level  ##########
    # punctation marks
    punctationMarkList = [\
	('<punctation-mark type="comma"></punctation-mark>'           , ''),\
	('<punctation-mark type="period"></punctation-mark>'          , ''),\
	('<punctation-mark type="question mark"></punctation-mark>'   , ''),\
	('<punctation-mark type="exclamation mark"></punctation-mark>', '')\
	]


    # ########## phrase level  ##########
    # verbal deletions
    verbalDeletionList = [
	('<verbally-deleted type="correction">' , ''),
	('</verbally-deleted>'                  , ''),
	('<verbally-deleted type="false start">', ''),
	('</verbally-deleted>'                  , '')
	]

    # proper nouns
    properNounList = [
	(re.compile(r'(?<= )<name> (\S+) </name>(?= )')          , r' ~\1 '),
	(re.compile(r'(?<= )<neologism> (\S+) </neologism>(?= )'), r' *\1 ')
	]


    # phrase
    phraseList = [
	(re.compile(r'(?<= )<phrase> ([^<]+) </phrase>(?= )'), lambda m: m.group(1).replace(' ', '_'))
	 ]


    # ########## word level  ##########
    prosodyList = [
	(re.compile(r'<prosody([^>]*)> \[hesitation\] +'), r'[hesitation]'),
	(re.compile(r'<prosody([^>]*)> +')               , r''            ),
	(re.compile(r' +\[hesitation\] </prosody>')      , r'[hesitation]'),
	(re.compile(r' +</prosody>')                     , r''            ),
	('[hesitation]'                                  , r'&lt;L&gt;'         )
	]

    # language
    languageList = [
	(re.compile(r'(?<= )<language lang="([^"]+)"> ([^<]+) </language>(?= )'),
	 lambda m: '&lt;*' + m.group(1) + '&gt;' + m.group(2).replace(' ', ' &lt;*' + m.group(1) + '&gt;'))
	]

    # spelled
    spelledList = [
	(re.compile(r'(?<= )<spelled> ([^<]+) </spelled>(?= )'),
	 lambda m: '$' + m.group(1).replace(' ', ' $'))
	]

    # breaks
    breakList = [
	(re.compile(r'(?<= )<broken cause="articulatory" location="end missing"> ([^<]+) </broken>(?= )')  , r'\1='   ),
	(re.compile(r'(?<= )<broken cause="articulatory" location="begin missing"> ([^<]+) </broken>(?= )'), r'_\1'   ),
	(re.compile(r'(?<= )<broken cause="technical" location="end missing"> ([^<]+) </broken>(?= )')     , r'&lt;_T&gt;\1'),
	(re.compile(r'(?<= )<broken cause="technical" location="begin missing"> ([^<]+) </broken>(?= )')   , r'&lt;T_&gt;\1'),
	('<recording-interrupted></recording-interrupted>'                                                 , r'&lt;*T&gt;'  )
	]

    # intelligibility
    intelligibilityList = [
	(re.compile(r'(?<= )<hardly-intelligible> ([^<]+) </hardly-intelligible>(?= )'), r'\1%'),
	('<unintelligible> [???] </unintelligible>'                                    , r'&lt;%&gt;')
	]

    #  noise
    noiseList = [
	(re.compile(r'(?<= )<hesitation[^>]*> ([^<]+) </hesitation>(?= )'),
	 lambda m: m.group(1).replace('[', '&lt;').replace(']', '&gt;')),
	(re.compile(r'(?<= )<noise[^>]*> ([^<]+) </noise>(?= )'),
	 lambda m: m.group(1).replace('[', '&lt;').replace(']', '&gt;'))
	]

    # escape xml-characters
    escapeCharList = [
	('&', '&amp;'),
#        ('<', '&lt;'),
#        ('>', '&gt;')
	]
    unEscapeCharList = [
	('&amp;', '&'),
	('&lt;' , '<'),
	('&gt;' , '>')
	]

    # warn, if xml-tags remained
    warnRE = re.compile(r'(?:</.*?>)|(?:<.*?/>)')

    def __init__(self, directory):
	if directory == '-':
	    self.directory = '-'
	else:
	    self.directory = os.path.abspath(directory)
	    if not os.path.exists(directory):
		os.makedirs(directory)
	self.setWarnList( [ self.warnRE ] )
	self.latin12ascii = ToAsciiConverter()

    def bliss2trl(self, s):
	s = self.normalize(s)

	# transformation on a word level
	# order is crucial !!!
	s = self.transform(s, self.noiseList)
	s = self.transform(s, self.prosodyList)
	s = self.transform(s, self.spelledList)
	s = self.transform(s, self.languageList)
	s = self.transform(s, self.breakList)
	s = self.transform(s, self.intelligibilityList)
	s = self.transform(s, self.verbalDeletionList)

	# transformation on a phrase level
	s = self.transform(s, self.phraseList)
	s = self.transform(s, self.properNounList)

	# transformation on a sentence level
	s = self.transform(s, self.punctationMarkList)

	return s


    def getOrthTagAttributes(self, attr):
	convAttr = dict([x for x in attr.items()])
	convAttr['format'] = 'trl'
	return convAttr

    def getPath(self, path):
	if self.directory == '-':
	    return '-'
	else:
	    return self.directory + '/trl_' + os.path.basename(path)

    def getOrthography(self, orth):
	origOrth = orth
	orth = self.transform(orth, self.escapeCharList)
	# fix point iteration
	lastOrth = orth
	orth = self.bliss2trl(lastOrth)
	while not lastOrth == orth:
	    lastOrth = orth
	    orth = self.bliss2trl(lastOrth)
	orth = self.squeeze(orth)
	self.warn(orth, origOrth)
	orth = self.transform(orth, self.unEscapeCharList)
	return orth




class OldConverter(BlissOrthographyConverter):
    """
    classes used to convert a bliss orthography to old systems orthography
    """

    # warn, if xml-tags remained
    warnRE = re.compile(r'(?:</.*?>)|(?:<.*?/>)')

    def __init__(self, directory):
	if directory == '-':
	    self.directory = '-'
	else:
	    self.directory = os.path.abspath(directory)
	    if not os.path.exists(directory):
		os.makedirs(directory)
	self.setWarnList( [ self.warnRE ] )
	self.ascii = ToAsciiConverter()


    def getPath(self, path):
	if self.directory == '-':
	    return '-'
	else:
	    return self.directory + '/old_' + os.path.basename(path)

    def getOrthTagAttributes(self, attr):
	convAttr = dict([x for x in attr.items()])
	convAttr['format'] = 'trl'
	return convAttr

    def getOrthography(self, s):
	o = s
	tokenList = self.split(s)
	wordList = []
	phraseList = []
	isPhrase = False
	for token in tokenList:
	    if token.startswith('<phrase'):
		isPhrase = True
	    elif token == '</phrase>':
		wordList.append('_'.join(phraseList))
		phraseList = []
		isPhrase = False
	    elif not token.startswith('<'):
		if isPhrase:
		    if not token.startswith('['):
			phraseList.append(token)
		else:
		    wordList.append(token)
	s = self.ascii.encode(' '.join(wordList))
	self.warn(s, o)
	return s



class IdConverter(BlissOrthographyConverter):
    """
    class implementing the id mapping
    """
    def __init__(self, directory):
	if directory == '-':
	    self.directory = '-'
	else:
	    self.directory = os.path.abspath(directory)
	    if not os.path.exists(directory):
		os.makedirs(directory)

    def getPath(self, path):
	if self.directory == '-':
	    return '-'
	else:
	    return self.directory + '/' + os.path.basename(path)

    def getOrthTagAttributes(self, attr):
	return attr

    def getOrthography(self, s):
	return s


class StmConverter(BlissOrthographyConverter):
    """
    class converting bliss to stm format
    """
    def __init__(self, target):
	if target == '-':
	    self.target = '-'
	else:
	    self.target = os.path.abspath(target)

    def getPath(self, path):
	return self.target

    def getOrthTagAttributes(self, attr):
	return attr

    def getOrthography(self, s, start="unk", end="unk", rec="unk"):
	speaker = "unknown"
	if end == "inf": end = 999999.9
	return "%s 1 %s %s %s <o,f0,unknown> %s\n" % (rec, speaker, start, end, s)


class SriConverter(BlissOrthographyConverter):
    """
    class converting bliss orthography to sri language model data orthography
    """
    def __init__(self, target):
	if target == '-':
	    self.target = '-'
	else:
	    self.target = os.path.abspath(target)
	self.isSplitAtSentence = False
	self.phraseConcatenator = ' '
	self.conv = LmIdTokenGenerator()
	self.sl = []
	self.wl = []

    def setSplitAtSentence(self, isSplitAtSentence):
	self.isSplitAtSentence = isSplitAtSentence
    def getSplitAtSentence(self):
	return self.isSplitAtSentence

    def setPhraseDelimiter(self, phraseConcatenator):
	self.phraseConcatenator = phraseConcatenator
    def getPhraseDelimiter(self):
	return self.phraseConcatenator

    def setLmTokenGenerator(self, conv):
	self.conv = conv
    def getLmTokenGenerator(self):
	return self.conv

    def getPath(self, path = None):
	if not path or self.target == '-':
	    return self.target
	else:
	    path = os.path.basename(path)
	    if path.endswith('.gz'):
		path = path[:-3]
		ext = '.sri.gz'
	    else:
		ext = '.sri'
	    if path.endswith('.corpus'):
		path = path[:-7]
	    return self.target + '/' + path + ext

    def join(self, wl):
	return self.phraseConcatenator.join(wl)

    def hasPrefixIn(self, s, pl):
	for p in pl:
	    if s.startswith(p):
		return True
	return False

    def convert(self, o):
	return ' '.join(self.conv.syntTokenSeq(o))

    def getOrthography(self, s):
	fl = []
	discard = False
	phrase = False
	sentenceEnd = False
	for w in self.split(s.strip()):
	    if w == '<phrase>':
		phrase = True
	    elif w == '</phrase>':
		phrase = False
		if fl:
		    self.wl.append(self.join(fl))
		    fl = []
	    elif w.startswith('<punctation-mark'):
		if w.startswith('<punctation-mark type="period"') \
		       or w.startswith('<punctation-mark type="question mark"') \
		       or w.startswith('<punctation-mark type="exclamation mark"'):
		    if self.wl:
			self.sl.append('<s> ' + ' '.join(self.wl) + ' </s>')
			self.wl = []
		if not w.endswith('/>'):
		    discard = True
	    elif w == '</punctation-mark>':
		discard = False
	    elif self.hasPrefixIn(w, ['<noise', '<hesitation']):
		discard = True
	    elif self.hasPrefixIn(w, ['</noise', '</hesitation']):
		discard = False
	    elif not discard and not self.hasPrefixIn(w, ['<', '[']):
		w = self.convert(w)
		if phrase:
		    fl.append(w)
		else:
		    self.wl.append(w)
	if self.wl and not self.isSplitAtSentence:
	    self.sl.append('<s> ' + ' '.join(self.wl) + ' </s>')
	    self.wl = []
	if self.sl:
	    s = '\n'.join(self.sl) + '\n'
	    self.sl = []
	else:
	    s = ''
	return s

    def getFinal(self):
	if self.wl:
	    s = '<s> ' + ' '.join(self.wl) + ' </s>\n'
	    self.wl = []
	else:
	    s = ''
	return s


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

    # parse
##    def includeSubcorpus(self, file):
##	if not file[0] == '/':
##	    if file.startswith('./'):
##		file = file[2:]
##	    file = self.base + '/' + file
##	self.parseList.append(file)

##    def parse(self, fileList):
##	if type(fileList) == type(''):
##	    fileList = [ fileList ]
##        for corpusFile in fileList:
##	    if corpusFile[0] not in './':
##		corpusFile = './' +  corpusFile
##            self.parseList.append(corpusFile)
##            self.startCorpus()
##            while self.parseList:
##                file = self.parseList.pop(0)
##                self.base, self.name = os.path.split(file)
##                self.startSubcorpus(file)
##                fd = zopen(file, 'r')
##                self.parser.parse(fd)
##                zclose(fd)
##                print '%-48s -->' % file
##                self.endSubcorpus()
##            self.endCorpus()


    def parseFile(self, filename) :
	parser = sax.make_parser()
	parser.setFeature(sax.handler.feature_namespaces, 0)
	parser.setFeature(sax.handler.feature_external_ges, False)
	parser.setFeature(sax.handler.feature_external_pes, False)
	parser.setContentHandler(self)
	fd = zopen(filename, 'r')
	parser.parse(fd)
	zclose(fd)
	print '%-48s -->' % filename

    def parseInclude(self, filename):
	if not filename[0] == '/':
	    if filename.startswith('./'):
		filename = filename[2:]
	    filename = self.base + '/' + filename
	self.parseFile(filename)

    def parseMain(self, filename):
	if filename[0] not in './':
	    filename = './' +  filename
	self.base = os.path.dirname(filename)
	self.parseFile(filename)

    def parse(self, fileList):
	if type(fileList) is str:
	    fileList = [ fileList ]
	for filename in fileList:
	    self.startMain(filename)
	    self.parseMain(filename)
	    self.endMain()



class BlissOrthographyConverterWriter(BlissCorpusParser):
    """
    reads a bliss corpus file, converts the orthography, and
    writes the result to file;
    the procedure is done recursively for all contained subcorpora
    """
    def __init__(self, encoding, converter, isSingleFile = False):
	BlissCorpusParser.__init__(self)
	#encoding
	self.encoding = encoding
	# converter
	self.conv = converter
	# writer
	self.filename = None
	self.writer = None
	self.isSingleFile = isSingleFile
	if not self.isSingleFile:
	    self.writerStack = []

    # writer
    def orthElement(self, attr, content):
	self.writer.write(self.conv.getOrthography(content))

    def startInclude(self, attr):
	filename = attr['file']
	if not self.isSingleFile:
	    self.writerStack.append( (self.filename, self.writer) )
	    self.filename =  self.conv.getPath(filename)
	    self.writer = uopen(self.filename, self.encoding, 'w')

    def endInclude(self):
	if not self.isSingleFile:
	    uclose(self.writer)
	    print '--> %-48s' % self.filename
	    self.filename, self.writer = self.writerStack.pop()

    def startMain(self, filename):
	if not self.isSingleFile:
	    self.filename = self.conv.getPath(filename)
	else:
	    self.filename = self.conv.getPath()
	self.writer  = uopen(self.filename, self.encoding, 'w')

    def endMain(self):
	uclose(self.writer)
	print '--> %-48s' % self.filename
	self.filename = None


class BlissOrthographyConverterStmWriter(BlissOrthographyConverterWriter):
    """
    reads a bliss corpus file, converts the orthography, and
    writes the result to stm file
    """
    def __init__(self, encoding, converter, isSingleFile = False):
	BlissOrthographyConverterWriter.__init__(self, encoding, converter,
						 isSingleFile)

    def startRecording(self, attr):
	self.cur_rec = attr['name']

    def startSegment(self, attr):
	self.cur_start = attr['start']
	self.cur_end   = attr['end']

    def orthElement(self, attr, content):
	line = self.conv.getOrthography(content, self.cur_start,
					self.cur_end, self.cur_rec)
	self.writer.write(line)


class BlissOrthographyConverterRewriter(BlissCorpusParser):
    """
    reads a bliss corpus file, converts the orthography, and
    writes the converted file (still in the bliss corpus format)
    the procedure is done recursively for all contained subcorpora
    """
    def __init__(self, encoding, converter, isSingleFile=False):
	BlissCorpusParser.__init__(self)
	# converter
	self.conv = converter
	self.encoding = encoding
	# writer
	self.filename = None
	self.xmlWriter = None
	self.isSingleFile = isSingleFile
	if not self.isSingleFile:
	    self.xmlWriterStack = []

    # misc.
    escapeSeq = [
	('&', '&amp;'),
	('<', '&lt;'),
	('>', '&gt;')
	]
    def escape(self, orth):
	for w, e in self.escapeSeq:
	    orth = orth.replace(w, e)
	return orth

    # xml (re)writer
    def begin(self):
	self.xmlWriter.openComment()
	self.xmlWriter.cdata('generated by ' + sys.argv[0])
	# self.writer.cdata(revision)
	self.xmlWriter.closeComment()

    def open(self, cdata, name, attr):
	if len(cdata) > 0:
	    self.xmlWriter.cdata(cdata)
	if self.dataOnlyElement:
	    self.xmlWriter.open(self.dataOnlyElementName, self.dataOnlyElementAttr)
	else:
	    self.dataOnlyElement = True
	self.dataOnlyElementName = name
	self.dataOnlyElementAttr = attr

    def close(self, cdata, name):
	if self.dataOnlyElement:
	    if len(cdata) == 0:
		self.xmlWriter.empty(self.dataOnlyElementName, self.dataOnlyElementAttr)
	    elif len(self.dataOnlyElementAttr) == 0 and cdata.find('\n') == -1:
		self.xmlWriter.element(self.dataOnlyElementName, cdata)
	    else:
		self.xmlWriter.open(self.dataOnlyElementName, self.dataOnlyElementAttr)
		self.xmlWriter.cdata(cdata)
		self.xmlWriter.close(name)
	    self.dataOnlyElement = False
	else:
	    if len(cdata) > 0:
		self.xmlWriter.cdata(cdata)
	    self.xmlWriter.close(name)

    def comment(self, cdata):
	self.xmlWriter.openComment()
	self.xmlWriter.cdata(cdata)
	self.xmlWriter.closeComment()


    # xml handler
    def startDocument(self):
	self.content = ''
	self.isOrth = False
	self.dataOnlyElement = False
	self.dataOnlyElementName = ''
	self.dataOnlyElementAttr = ''

    def endDocument(self):
	pass

    def startElement(self, name, attr):
	if self.isOrth:
	    self.content += "<" + ' '.join([name] + map(lambda kv: '%s="%s"' % kv, attr.items())) + ">"
	else:
	    if name == "orth":
		attr = self.conv.getOrthTagAttributes(attr)
		self.isOrth = True
	    elif name == "include":
		filename = attr['file']
		attr = dict(attr)
		self.parseSubcorpus(filename)
		attr['file'] = self.conv.getPath(filename)
	    self.open(self.escape(self.cdata()), name, attr)

    def endElement(self, name):
	if self.isOrth:
	    if name == 'orth':
		self.close(self.escape(self.conv.getOrthography(self.cdata())), 'orth')
		#self.comment(cdata)
		self.isOrth = False
	    else:
		self.content += "</" + name + ">"
	else:
	    self.close(self.escape(self.cdata()), name)


    # bliss corpus handler
    def startInclude(self, attr):
	filename = attr['file']
	if not self.isSingleFile:
	    self.xmlWriterStack.append( (self.filename, self.xmlWriter) )
	    self.filename =  self.conv.getPath(filename)
	    self.xmlWriter = openXml(self.filename, self.encoding)

    def endInclude(self):
	if not self.isSingleFile:
	    closeXml(self.xmlWriter)
	    print '--> %-48s' % self.filename
	    self.filename, self.xmlWriter = self.xmlWriterStack.pop()

    def startMain(self, filename):
	if not self.isSingleFile:
	    self.filename = self.conv.getPath(filename)
	else:
	    self.filename = self.conv.getPath()
	self.xmlWriter  = openXml(self.filename, self.encoding)

    def endMain(self):
	closeXml(self.xmlWriter)
	print '--> %-48s' % self.filename
	self.filename = None



class WordListExtractor(BlissCorpusParser):
    """
    reads a bliss corpus file, converts the orthography, and
    writes the result to file;
    the procedure is done recursively for all contained subcorpora
    """
    def __init__(self, encoding, vocabBase):
	BlissCorpusParser.__init__(self)
	self.encoding = encoding
	self.vocabBase = vocabBase
	self.concatenatePhrases = True
	self.phraseDelimiter = ' '

    def setPhrases(self, concatenatePhrases):
	self.concatenatePhrases = concatenatePhrases

    def setPhraseDelimiter(self, phraseDelimiter):
	self.phraseDelimiter = phraseDelimiter

    def getPhraseDelimiter(self):
	return self.phraseDelimiter

    def startMain(self, filename) :
	base, name = os.path.split(filename)
	if name.endswith('.gz'):
	    name = name[:-3]
	if name.endswith('.corpus'):
	    name = name[:-7]
	self.vocabBase = base
	self.vocabFileName = name
	self.wordCount = {}
	self.classMap = {}
	self.stack = []
	self.isPhrase = False

    def endMain(self):
	countList = [(-count, word) for word, count in self.wordCount.iteritems()]
	countList.sort()
	fileOut = self.vocabBase + '/' + self.vocabFileName + '.vocab.count'
	fd = uopen(fileOut, self.encoding, 'w')
	for count, word in countList:
	    print >> fd, str(-count) + ' \t' + word
	uclose(fd)
	print '--> %-48s' % fileOut

	wordList = self.wordCount.keys()
	wordList.sort()
	fileOut = self.vocabBase + '/' + self.vocabFileName + '.vocab'
	fd = uopen(fileOut, self.encoding, 'w')
	for word in wordList:
	    print >> fd, word
	uclose(fd)
	print '--> %-48s' % fileOut

	wordSet = set(wordList)
	for tag, tagWordSet in self.classMap.iteritems():
	    wordSet -= tagWordSet
	    wordList = list(tagWordSet)
	    wordList.sort()
	    fileOut = self.vocabBase + '/' + self.vocabFileName + '.' + tag + '.vocab'
	    fd = uopen(fileOut, self.encoding, 'w')
	    for word in wordList:
		print >> fd, word
	    zclose(fd)
	    print '--> %-48s' % fileOut
	wordList = list(wordSet)
	wordList.sort()
	fileOut = self.vocabBase + '/' + self.vocabFileName + '.orth.vocab'
	fd = uopen(fileOut, self.encoding, 'w')
	for word in wordList:
	    print >> fd, word
	uclose(fd)
	print '--> %-48s' % fileOut


    def count(self, w):
	self.wordCount[w] = self.wordCount.setdefault(w, 0) + 1
	for tag in self.stack:
	    self.classMap.setdefault(tag, set()).add(w)


    # splits a bliss-xml sentence
    # example: '<tag attr=".."> [some noise] text </tag>' -->  ['<tag attr="..">', '[some noise]', 'text', '</tag>']
    blissTokenRE = re.compile(r'(?:<.+?>)|(?:\[.+?\])|(?:\S+)')
    def split(self, s):
	return self.blissTokenRE.findall(s)

    langRE = re.compile(r'<language\s+lang\s*=\s*"([^"]+)"\s*>')
    def orthElement(self, attr, s):
	pl = []
	discard = False
	for w in self.split(s.strip()):
	    if w == '</phrase>':
		if self.concatenatePhrases:
		    self.count(self.phraseDelimiter.join(pl))
		    self.isPhrase = False
	    elif w.startswith('</'):
		self.stack.pop()
	    elif w.startswith('<phrase'):
		if not w.endswith('/>'):
		    if self.concatenatePhrases:
			self.isPhrase = True
	    elif w.startswith('<language'):
		if not w.endswith('/>'):
		    self.stack.append("lang-" + self.langRE.match(w).group(1))
	    elif w.startswith('<'):
		if not w.endswith('/>'):
		    i = w.find(' ')
		    if i > 0:
			self.stack.append(w[1:i])
		    else:
			self.stack.append(w[1:-1])
	    else:
		if self.isPhrase:
		    pl.append(w)
		else:
		    self.count(w)


class SyncSegmenter:
    """
    classes used to create a bliss corpus
    used format: list of triples (start-time, end-time, orth)
    Segmenter-Classes
    """
    def getSegmentList(self, segments):
	return segments

class PunctationSegmenter:
    def __init__(self, punctationMarks = '.?!'):
	self.punctationMarks = punctationMarks

    def isSentenceEndSymbol(self, orth):
	return  orth[-1] in self.punctationMarks

    def getSegmentList(self, segments):
	blissSegments = []
	rows = []
	lastStart, lastEnd = segments[0][0], segments[0][1]
	for start, end, orth in segments:
	    if start > lastEnd:
		blissSegments.append( (lastStart, lastEnd, '\n'.join(rows)) )
		lastStart, rows = start, []
	    rows.append(orth)
	    orth = orth.strip()
	    if orth:
		if self.isSentenceEndSymbol(orth):
		    blissSegments.append( (lastStart, end, '\n'.join(rows)) )
		    lastStart, rows = end, []
	    lastEnd = end
	if rows:
	    blissSegments.append( (lastStart, end, '\n'.join(rows)) )
	return blissSegments



# ******************************************************************************
# ==> audio and out path must be setable from outside

# output filter
class AbstractFilter:
    name     = ''
    prefix   = ''
    outDir   = ''
    audioDir = ''
    audioExt = ''

    def __init__(self, name = ''):
	if name:
	    self.name = name
	    self.prefix = name + '.'

    def _normalizeDir(self, dir):
	if dir and not dir.endswith('/'):
	    return dir + '/'
	else:
	    return dir

    def _normalizeExt(self, ext):
	if ext.startswith('.'):
	    return ext
	else:
	    return '.' + ext

    def setOutputDirectory(self, outDir):
	self.outDir = self._normalizeDir(outDir)

    def getOutputDirectory(self):
	return self.outDir

    def setAudioDirectory(self, audioDir):
	self.audioDir = self._normalizeDir(audioDir)

    def setAudioExtension(self, audioExt):
	self.audioExt = self._normalizeExt(audioExt)


    # directory structure
    # audio
    # 1) return rec.audioFile, if not set
    # 2) build name of default audio dir, default audio ext, and rec name, if default audio dir is not set
    # 3) return relative dir, i.e. ../audio
    def makeAudioPath(self, rec):
	if rec.audioFile:
	    return rec.audioFile
	else:
	    return self.audioDir + rec.name + self.audioExt

    # subcorpora
    # 1) return subcorpus.file, if not set
    # 2) build name of default base dir and subcorpus name, if default out dir is not set
    # 3) return relative dir, i.e. ./
    def makeSubCorpusPath(self, subcorpus):
	if subcorpus.file:
	    return subcorpus.file
	else:
	    return self.outDir + self.prefix + subcorpus.name + '.corpus.gz'

    # speaker, and condition corpus
    # 1) return path, if not set
    # 2) return default corpus name
    def makeSpeakerCorpusPath(self, path = None):
	if path:
	    return path
	else:
	    return 'speaker.corpus.gz'

    def makeConditionCorpusPath(self, path = None):
	if path:
	    return path
	else:
	    return 'condition.corpus.gz'

    # include corpus
    # 1)  return path, if not set
    # 2) return relative dir, i.e. ./, if name is set
    # 3) return None, if name is not set
    def makeIncludeCorpusPath(self, path = None):
	if path:
	    return path
	elif self.prefix:
	    return self.prefix + 'corpus.gz'
	else:
	    return None

    # abstract function
    def match(self, seg):
	raise NotImplemented



class IdFilter(AbstractFilter):
    def __init__(self, name):
	AbstractFilter.__init__(self, name)
	self.outDir   = 'complete/'
	self.audioDir = '../../audio/'

    def match(self, seg):
	return True


class GenderFilter(AbstractFilter):
    def __init__(self, name, gender, corpusParser):
	AbstractFilter.__init__(self, name)
	self.outDir   = 'byGender/'
	self.audioDir = '../../audio'
	self.gender = gender
	self.corpusParser = corpusParser
	self.kvPair = ('gender', gender)
	self.prefix += gender + '.'

    def match(self, seg):
	return self.kvPair in self.corpusParser.spkList[seg.spkId]


class SpeakerFilter(AbstractFilter):
    ascii = ToAsciiConverter()

    def __init__(self, name, spkId, corpusParser):
	AbstractFilter.__init__(self, name)
	self.outDir   = 'bySpeaker/'
	self.audioDir = '../../audio'
	self.spkId = spkId
	self.spkDesc = 'spk' + str(spkId)
	for k, v in corpusParser.spkList[spkId]:
	    if k == 'name':
		self.spkExtDesc =  str(spkId) + '-' + \
				  self.ascii.encode('_'.join(v.split()))
		break
	else:
	    self.spkExtDesc = self.spkDesc
	self.prefix += self.spkExtDesc + '.'

    def match(self, seg):
	return self.spkId == seg.spkId


class ConditionFilter(AbstractFilter):
    def __init__(self, name, condId, corpusParser):
	AbstractFilter.__init__(self, name)
	self.outDir   = 'byCondition/'
	self.audioDir = '../../audio'
	self.condId = condId
	self.condDesc = 'cond' + str(condId)
	self.prefix += self.condDesc + '.'

    def match(self, seg):
	return self.condId == seg.condId


# ******************************************************************************


# corpus elements
class Subcorpus:
    __slots__ = ('name', 'file', 'recList')
    def __init__(self, name, file, recList):
	self.name, self.file, self.recList = name, file, recList

class Recording:
    __slots__ = ('name', 'audioFile', 'segList')
    def __init__(self, name, audioFile, segList):
	self.name, self.audioFile, self.segList = name, audioFile, segList

class Segment:
    __slots__ = ('spkId', 'condId', 'start', 'end', 'orth', 'foreignOrth')
    def __init__(self, spkId, condId, start, end, orth, foreignOrth = tuple()):
	self.spkId, self.condId, self.start, self.end, self.orth, self.foreignOrth = \
		    spkId, condId, start, end, orth, foreignOrth


# ******************************************************************************


# main class
class CorpusParser:
    def __init__(self, corpusName, encoding, baseDir = '.'):
	self.corpusName  = corpusName
	self.encoding = encoding
	self.fileParser  = None
	self.baseDir     = baseDir
	self.relOutDir   = None
	self.relAudioDir = None
	self.audioExt    = None
	self.hesitationSet  = set()
	self.noiseSet       = set()
	self.genderSet      = set()
	self.spkList  = []
	self.condList = []
	self.spkLookUp  = {}
	self.condLookUp = {}
	self.subcorpus = self._firstTimeSubcorpus
	self.recording = self._subcorpusAndRecording
	self.resetCorpus()

    def resetCorpus(self):
	# reset statistics
	self.noiseN      = 0
	self.hesitationN = 0
	self.foreignN    = 0
	self.nativeN     = 0
	self.maxTokenPerOrth = 0
	self.isDiscardEmptySegment   = True
	self.isDiscardNoiseSegment   = True
	self.isDiscardForeignSegment = True

	self.discardNoiseN      = 0
	self.discardHesitationN = 0
	self.discardForeignN    = 0
	self.discardNativeN    = 0

	self.discardEmptySegmentN   = 0
	self.discardNoiseSegmentN   = 0
	self.discardForeignSegmentN = 0
	self.discardByUserSegmentN = 0
	self.discardSegmentList = []

	self.sumSegmentDuration = 0.0
	# reset data structure
	self._newSubcorpusList()

    # modify behaviour
    def setDiscardEmptySegment(self, isDiscardEmptySegment = True):
	self.isDiscardEmptySegment = isDiscardEmptySegment
    def setDiscardNoiseSegment(self, isDiscardNoiseSegment = True):
	self.isDiscardNoiseSegment = isDiscardNoiseSegment
    def setDiscardForeignSegment(self, isDiscardForeignSegment = True):
	self.isDiscardForeignSegment = isDiscardForeignSegment

    # overwrite this, to set your own discard rules
    def discardSegment(self, seg):
	return False

    def setFileParser(self, fileParser):
	fileParser.corpusParser = self
	if fileParser.converter:
	    fileParser.converter.corpusParser = self
	self.fileParser = fileParser
    def getFileParser(self):
	return self.fileParser

    def setSegmenter(self, segmenter):
	self.fileParser.segmenter = segmenter
    def getSegmenter(self):
	return self.fileParser.segmenter

    def setConverter(self, converter):
	converter.corpusParser = self
	self.fileParser.converter = converter
    def getConverter(self):
	return self.fileParser.converter


    def setBaseDirectory(self, baseDir):
	if baseDir == '':
	    baseDir = '.'
	self.baseDir = os.path.realpath(baseDir) + '/'

    # directory relative to base directory (see above),
    # absolute pathes are allowed
    def setRelativeSubcorpusDirectory(self, relOutDir):
	self.relOutDir = relOutDir

    # directory relative to base directory (see above)
    # absolute pathes are allowed
    def setRelativeAudioDirectory(self, relAudioDir):
	self.relAudioDir = relAudioDir

    def setAudioExtension(self, audioExt):
	self.audioExt = audioExt

    # parse
    def parse(self, fileList):
	if type(fileList) == type(''):
	    fileList = [ fileList ]
	for file in fileList:
	    self.fileParser.parse(file)

    # misc
    def _makeAbsolutePath(self, path):
	if path.startswith('/'):
	    return os.path.realpath(path)
	else:
	    return os.path.realpath(self.baseDir + '/' + path)


    def _mkdirs(self, path):
	path = os.path.dirname(path)
	if not os.path.exists(path):
	    os.makedirs(path)

    def _setFilterDirs(self, filter):
	if self.relOutDir is not None:
	    filter.setOutputDirectory(self.relOutDir)
	if self.relAudioDir is not None:
	    filter.setAudioDirectory(self.relAudioDir)
	if self.audioExt is not None:
	    filter.setAudioExtension(self.audioExt)

    # analyzes the orthography of a segment,
    # use result for some corpora statistics
    blissTokenRE = re.compile(r'(?:<.+?>)|(?:\[.+?\])|(?:\S+)')
    def _analyzeSegment(self, seg):
	tagN, noiseN, hesitationN, foreignN, nativeN = 0, 0, 0, 0, 0
	isNoise,isHesitation, isForeign = False, False, False
	for t in self.blissTokenRE.findall(seg.orth):
	    if t[0] == '<':
		tagN += 1
		if t.startswith('<noise'):
		    isNoise = True
		elif t == '</noise>':
		    isNoise = False
		elif t.startswith('<hesitation'):
		    isHesitation = True
		elif t == '</hesitation>':
		    isHesitation = False
		elif t.startswith('<language'):
		    isForeign = True
		elif t == '</language>':
		    isForeign = False
	    else:
		if isNoise:
		    self.noise(t)
		    noiseN += 1
		elif isHesitation:
		    self.hesitation(t)
		    hesitationN += 1
		elif isForeign:
		    foreignN += 1
		else:
		    nativeN += 1
	tokenN = noiseN + hesitationN + foreignN + nativeN

	if self.isDiscardEmptySegment and tokenN == 0:
	    self.discardEmptySegmentN += 1
	    self.discardSegmentList.append(seg)
	    return False
	if self.isDiscardNoiseSegment and noiseN == tokenN > 0:
	    self.discardNoiseSegmentN += 1
	    self.discardNoiseN += noiseN
	    self.discardSegmentList.append(seg)
	    return False
	if self.isDiscardForeignSegment and foreignN >= 1 > nativeN:
	    self.discardForeignSegmentN += 1
	    self.discardNoiseN      += noiseN
	    self.discardHesitationN += hesitationN
	    self.discardForeignN    += foreignN
	    self.discardSegmentList.append(seg)
	    return False
	if self.discardSegment(seg):
	    self.discardByUserSegmentN += 1
	    self.discardNoiseN      += noiseN
	    self.discardHesitationN += hesitationN
	    self.discardForeignN    += foreignN
	    self.discardNativeN     += nativeN
	    self.discardSegmentList.append(seg)
	    return False

	self.noiseN      += noiseN
	self.hesitationN += hesitationN
	self.foreignN    += foreignN
	self.nativeN     += nativeN
	self.maxTokenPerOrth = max(self.maxTokenPerOrth, tokenN)
	return True


    # data structure
    def _newSubcorpusList(self):
	self._newRecordingList()
	self.subList = []

    def _newRecordingList(self):
	self._newSegmentList()
	self.recList = []

    def _newSegmentList(self):
	self.segList = []

    # follow flow of corpus/bliss file
    def _subcorpus(self, name, file = None):
	self._newRecordingList()
	self.subList.append( Subcorpus(name, file, self.recList) )

    def _firstTimeSubcorpus(self, name, file = None):
	 self.recording = self._recording
	 self.subcorpus = self._subcorpus
	 self._subcorpus(name, file)

    def _recording(self, name, audioFile = None):
	self._newSegmentList()
	self.recList.append( Recording(name, audioFile, self.segList) )

    def _subcorpusAndRecording(self, name, audioFile = None):
	self._subcorpus(name)
	self.recList.append( Recording(name, audioFile, self.segList) )

    def speaker(self, speaker):
	name = speaker.setdefault('name', 'unknown')
	gender = speaker.setdefault('gender', 'unknown')
	self.genderSet.add(speaker['gender'])
	speaker = speaker.items()
	speaker.sort()
	speaker = tuple(speaker)
	id = self.spkLookUp.get(speaker)
	if id is None:
	    id = len(self.spkList)
	    self.spkList.append(speaker)
	    self.spkLookUp[speaker] = id
	self.spkId = id
	return self.spkId
    def speakerId(self, spkId):
	self.spkId  = spkId

    def condition(self, condition):
	condition = condition.items()
	condition.sort()
	condition = tuple(condition)
	id = self.condLookUp.get(condition)
	if id is None:
	    id = len(self.condList)
	    self.condList.append(condition)
	    self.condLookUp[condition] = id
	self.condId = id
	return self.condId
    def conditionId(self, condId):
	self.condId = condId

    def noise(self, noiseSymbol):
	self.noiseSet.add(noiseSymbol)

    def hesitation(self, hesiationSymbol):
	self.hesitationSet.add(hesiationSymbol)


    def segment(self, start, end, orth, foreignOrth = tuple()):
	orth = orth.strip()
	seg = Segment(self.spkId, self.condId, start, end, orth, foreignOrth)
	# ATTENTION: analyzeSegment_ returns False, if segment should be discarded
	if self._analyzeSegment(seg):
	    self.segList.append(seg)
	    if start is None or end is None:
		self.sumSegmentDuration = float('nan')
	    else:
		self.sumSegmentDuration += end - start

    def _writeHeader(self, xml):
	xml.openComment()
	xml.cdata('created by ')
	xml.cdata(os.path.basename(sys.argv[0]))
	xml.closeComment()

    def _writeRecording(self, xml, rec, filter):
	if rec.segList:
	    if rec.audioFile:
		audioFile = rec.audioFile
	    else:
		audioFile = filter.makeAudioPath(rec)
	    xml.open('recording', name=rec.name, audio=audioFile)
	    for seg in rec.segList:
		self._writeSegment(xml, seg, filter)
	    xml.close('recording')

    def _writeSegment(self, xml, seg, filter):
	if seg.start is None or seg.end is None:
	    xml.open('segment')
	else:
	    xml.open('segment', start=seg.start, end=seg.end)
	xml.empty('speaker', name = str(seg.spkId))
	xml.empty('condition', name = str(seg.condId))
	xml.open('orth')
	xml.cdata(seg.orth)
	xml.close('orth')
	for lang, orth in seg.foreignOrth:
	    xml.openComment()
	    xml.open('trans', lang=lang)
	    xml.cdata(orth)
	    xml.close('trans')
	    xml.closeComment()
	xml.close('segment')

    def _dumpSet(self, path, s):
	self._mkdirs(path)
	fd = uopen(path, 'ascii', 'w')
	for e in s:
	    fd.write(e + '\n')
	uclose(fd)

    def _dumpSubCorpus(self, sub, filter):
	path = self._makeAbsolutePath(filter.makeSubCorpusPath(sub))
	self._mkdirs(path)
	xml = openXml(path, self.encoding)
	self._writeHeader(xml)
	xml.open('corpus', name=self.corpusName)
	for rec in sub.recList:
	    self._writeRecording(xml, rec, filter)
	xml.close('corpus')
	closeXml(xml)
	print 'sub       -->', path

    def _escape(self, cdata):
	return cdata.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def _dumpSpeakerCorpus(self, path, spkList, filter):
	path = self._makeAbsolutePath(filter.makeSpeakerCorpusPath(path))
	self._mkdirs(path)
	xml = openXml(path, self.encoding)
	xml.open('corpus', name=self.corpusName)
	for id, entry in enumerate(spkList):
	    xml.open('speaker-description', name=str(id))
	    for tag, cdata in entry:
		tag, cdata = self._escape(tag), self._escape(cdata)
		xml.open(tag)
		xml.cdata(cdata)
		xml.close(tag)
	    xml.close('speaker-description')
	xml.close('corpus')
	closeXml(xml)
	print 'speaker   -->', path

    def _dumpConditionCorpus(self, path, condList, filter):
	path = self._makeAbsolutePath(filter.makeConditionCorpusPath(path))
	self._mkdirs(path)
	xml = openXml(path, self.encoding)
	xml.open('corpus', name=self.corpusName)
	for id, entry in enumerate(condList):
	    xml.open('condition-description', name=str(id))
	    for tag, cdata in entry:
		tag, cdata = self._escape(tag), self._escape(cdata)
		xml.open(tag)
		xml.cdata(cdata)
		xml.close(tag)
	    xml.close('condition-description')
	xml.close('corpus')
	closeXml(xml)
	print 'condition -->', path


    def _dumpIncludeCorpus(self, path, subList, filter):
	path = filter.makeIncludeCorpusPath(path)
	if path:
	    path = self._makeAbsolutePath(path)
	    self._mkdirs(path)
	    xml = openXml(path, self.encoding)
	    self._writeHeader(xml)
	    xml.open('corpus', name=self.corpusName)
	    xml.empty('include', file=filter.makeSpeakerCorpusPath())
	    xml.empty('include', file=filter.makeConditionCorpusPath())
	    for sub in subList:
		xml.empty('include', file=filter.makeSubCorpusPath(sub))
	    xml.close('corpus')
	    closeXml(xml)
	    print 'include   -->', path


    def _dumpCorpus(self, path, subList, filter):
	self._dumpSpeakerCorpus(None, self.spkList, filter)
	self._dumpConditionCorpus(None, self.condList, filter)
	for sub in subList:
	    self._dumpSubCorpus(sub, filter)
	self._dumpIncludeCorpus(path, subList, filter)

    # dump
    def info(self, optionsList = ['discardedSegments']):
	tokenN = \
	       self.nativeN + self.foreignN + self.noiseN + self.hesitationN
	discardTokenN  = \
		      self.discardNativeN + self.discardForeignN + \
		      self.discardNoiseN + self.discardHesitationN
	discardSegmentN = \
			self.discardEmptySegmentN + self.discardNoiseSegmentN + \
			self.discardForeignSegmentN + self.discardByUserSegmentN
	if 'discardedSegments' in optionsList:
	    print 'discarded segments:'
	    for seg in self.discardSegmentList:
		print str(seg.start), str(seg.end), [ seg.orth ]
	s = int(self.sumSegmentDuration)
	m = s / 60; s -= m * 60
	h = m / 60; m -= h * 60
	duration = str(h) + 'h'
	if m < 10:
	    duration += '0 ' + str(m) + 'm'
	else:
	    duration += ' ' + str(m) + 'm'
	if s < 10:
	    duration += '0 ' + str(s) + 's'
	else:
	    duration += ' ' + str(s) + 's'

	print
	print '#token:'
	print 'token in native language   ', str(self.nativeN)
	print 'token in a foreign language', str(self.foreignN)
	print 'noise token                ', str(self.noiseN)
	print 'hesitation token           ', str(self.hesitationN)
	print '-----------------------------------'
	print '                           ', str(tokenN)
	print
	print '#discarded token:'
	print 'token in native language   ', str(self.discardNativeN)
	print 'token in a foreign language', str(self.discardForeignN)
	print 'noise token                ', str(self.discardNoiseN)
	print 'hesitation token           ', str(self.discardHesitationN)
	print '-----------------------------------'
	print '                           ', str(discardTokenN)
	print
	print '#discarded segments:'
	print 'empty segment              ', str(self.discardEmptySegmentN)
	print 'noise segment              ', str(self.discardNoiseSegmentN)
	print 'foreign segment            ', str(self.discardForeignSegmentN)
	print 'by user                    ', str(self.discardByUserSegmentN)
	print '-----------------------------------'
	print '                           ', str(discardSegmentN)
	print
	print 'max. number of token per segment:', str(self.maxTokenPerOrth)
	print 'sum up segment durations:', duration
	print

    def dumpNoise(self, path = None):
	if not path:
	    path = self._makeAbsolutePath('noise.lst')
	self._dumpSet(path, self.noiseSet)
	print 'noise -->', path

    def dumpHesitation(self, path = None):
	if not path:
	    path = self._makeAbsolutePath('hesitation.lst')
	self._dumpSet(path, self.hesitationSet)
	print 'hesitation -->', path

    def dumpSpeakerCorpus(self, path = None):
	filter = IdFilter('')
	self._setFilterDirs(filter)
	self._dumpSpeakerCorpus(path, self.spkList, filter)

    def dumpConditionCorpus(self, path = None):
	filter = IdFilter('')
	self._setFilterDirs(filter)
	self._dumpSpeakerCorpus(path, self.spkList, filter)

    def dumpSubCorpora(self):
	filter = IdFilter('')
	self._setFilterDirs(filter)
	for sub in subList:
	    self._dumpSubCorpus(sub, filter)

    def dumpIncludeCorpus(self, path):
	filter = IdFilter('')
	self._setFilterDirs(filter)
	self._dumpSpeakerCorpus(path, self.subList, filter)

    def dumpCorpus(self, name = '', includeCorpusPath = None):
	filter = IdFilter(name)
	self._setFilterDirs(filter)
	self._dumpCorpus(includeCorpusPath, self.subList, filter)

    def dumpFilteredCorpus(self, filter, includeCorpusPath = None):
	self._setFilterDirs(filter)
	subList = []
	for sub in self.subList:
	    recList = []
	    for rec in sub.recList:
		segList = [ seg for seg in rec.segList if filter.match(seg) ]
		if segList:
		    recList.append( Recording(rec.name, rec.audioFile, segList) )
	    if recList:
		subList.append( Subcorpus(sub.name, sub.file, recList) )
	if subList:
	    self._dumpCorpus(includeCorpusPath, subList, filter)


    def dumpCorpusByGender(self, name = '', includeCorpusPath = None):
	for gender in self.genderSet:
	    filter = GenderFilter(name, gender, self)
	    self.dumpFilteredCorpus(filter, includeCorpusPath)

    def dumpCorpusBySpeaker(self, name = '', includeCorpusPath = None):
	for id in range(len(self.spkList)):
	    filter = SpeakerFilter(name, id, self)
	    self.dumpFilteredCorpus(filter, includeCorpusPath)

    def dumpCorpusByCondition(self, name = '', includeCorpusPath = None):
	for id in range(len(self.condList)):
	    filter = ConditionFilter(name, id, self)
	    self.dumpFilteredCorpus(filter, includeCorpusPath)

    def dumpSingleFile(self, name = '-', extractGender = True):
	if name != '-':
	    name = self._makeAbsolutePath(name)
	xml = openXml(name, self.encoding)

	xml.open('corpus', name=self.corpusName)
	for sub in self.subList:
	    for rec in sub.recList:
		xml.open('recording', name=rec.name, audio=rec.audioFile)
		for seg in rec.segList:
		    xml.open('segment', start=seg.start, end=seg.end)
		    if extractGender == True:
			xml.empty('speaker' , dict(self.spkList[seg.spkId]))
			xml.empty('condition', dict(self.condList[seg.condId]))
		    xml.open('orth')
		    xml.cdata(seg.orth)
		    xml.close('orth')
		    for lang, orth in seg.foreignOrth:
			xml.openComment()
			xml.open('orth', lang=lang)
			xml.cdata(orth)
			xml.close('orth')
			xml.closeComment()
		    xml.close('segment')
		xml.close('recording')
	xml.close('corpus')
	closeXml(xml)
	print '-->', name

    def dumpTrsFile(self, name = '-'):
	xml = openXml(name, self.encoding)
	xml.setIndent_str('')
	xml.setMargin(sys.maxint)
	xml.cdata('<!DOCTYPE Trans SYSTEM "trans-13.dtd">')

	for sub in self.subList:
	    for rec in sub.recList:
		recEnd = 0.0
		spkIdDict = {}
		for seg in rec.segList:
		    spkIdDict[seg.spkId] = 1
		    if recEnd < float(seg.end):
			recEnd = float(seg.end)

		xml.open('Trans', scribe='i6 RWTH Aachen', audio_filename=os.path.basename(rec.audioFile)[:-4], version_date='041217' ) #, version="4", version_date="041209")
		xml.open('Topics')
		xml.empty('Topic', id='to1', desc=self.corpusName)
		xml.close('Topics')
		xml.open('Speakers')
		for id, speaker in enumerate(self.spkList):
		    speakerDict = dict(speaker)
		    speakerName = speakerDict.get('name', 'unknown')
		    speakerGender = speakerDict.get('gender', 'unknown')
		    speakerScope = speakerDict.get('scope', 'unknown')
		    speakerDialect = speakerDict.get('dialect', 'unknown')
		    speakerAccent = speakerDict.get('accent', '')
		    if spkIdDict.has_key(id) and speakerDict.get('id') != 'unk':
			xml.empty('Speaker', id='spk'+str(id), name=speakerName, scope=speakerScope, dialect=speakerDialect,
				  accent=speakerAccent, type=speakerGender)
		xml.close('Speakers')
		xml.open('Episode')

		xml.open('Section', type='report', topic='to1', startTime='0.0', endTime=str(recEnd))
		spkId = None
		orthList = []
		for seg in rec.segList:
		    if seg.spkId != spkId:
			if spkId != None:
			    if dict(self.spkList[spkId]).get('id') != 'unk':
				xml.open('Turn', speaker='spk'+str(spkId), startTime=turnStart, endTime=turnEnd)
			    else:
				xml.open('Turn', startTime=turnStart, endTime=turnEnd)
			    for orth in orthList:
				xml.cdata(orth+'\n')
			    xml.close('Turn')
			    orthList = []
			spkId = seg.spkId
			turnStart = seg.start
		    turnEnd = seg.end
		    orthList += [ '<Sync time="'+str(seg.start)+'"/>', seg.orth]
		if spkId != None:
		    if dict(self.spkList[spkId]).get('id') != 'unk':
			xml.open('Turn', speaker='spk'+str(spkId), startTime=turnStart, endTime=turnEnd)
		    else:
			xml.open('Turn', startTime=turnStart, endTime=turnEnd)
		    for orth in orthList:
			xml.cdata(orth)
		    xml.close('Turn')
		xml.close('Section')
		xml.close('Episode')
		xml.close('Trans')
	closeXml(xml)
	print '-->', name


class TextRowParser:
    def __init__(self, audioDirName = '', audioExtension = '.wav', extractGender = False ):
	# controlled by corpus parser
	self.corpusParser = None
	self.segmenter = SyncSegmenter()
	self.converter = BlissOrthographyConverter()
	# handler
	self.audioFileName = None
	self.spkIdDict = {}
	self.audioDirName_ = audioDirName
	self.audioExtension_ = audioExtension
	self.extractGender_ = extractGender

     # modify behaviour
    def getRecordingName(self):
	return self.recordingName_

    def getAudioFileName(self):
	return self.audioDirName_+self.audioFileName_+self.audioExtension_

class StmFileParser(TextRowParser):
    # parse
    def parse(self, stmFileName):
	file  = uopen(stmFileName, self.corpusParser.encoding, 'r')
	lineList = file.readlines()
	uclose(file)

	previousRecording = None
	for line in lineList:
	    if not line.startswith(';;'):
		token = line.strip().split()
		self.audioFileName_ = token[0]
		self.recordingName_ = token[0]
		recordingChannel = token[1]
		speakerId = token[2]
		segmentStart = float(str(token[3]))
		segmentEnd = float(str(token[4]))
		subsetIds = token[5]
		orth = ' '.join( token[6:] )

		if orth.upper().find("IGNORE_TIME_SEGMENT_IN_SCORING"):
		    if previousRecording != self.getAudioFileName():
			previousRecording = self.getAudioFileName()
			self.corpusParser.recording(self.getRecordingName(), self.getAudioFileName() )
		    spkId = self.corpusParser.speaker( { 'name': speakerId } )
		    condId = self.corpusParser.condition( { 'name': subsetIds } )
		    self.corpusParser.segment(segmentStart, segmentEnd, self.converter.getOrthography(orth))

	print stmFileName, '-->'

class PemFileParser(TextRowParser):
    """
    parse
    The PEM ("partitioned evaluation map") file format is given in the SCLITE documentation available through NIST's web page (http://www.nist.gov/speech/software.htm). Each record contains 5 fields: < filename >, < channel ("A" or "B") >, < speaker ("unknown") >, < begin time > and < end time >.
    """
    def parse(self, pemFileName):
	file  = uopen(pemFileName, self.corpusParser.encoding, 'r')
	lineList = file.readlines()
	uclose(file)

	previousRecording = None
	for line in lineList:
	    if not line.startswith(';;'):
		token = line.strip().split()
		self.audioFileName_ = token[0]
		self.recordingName_ = token[0]
		recordingChannel = token[1]
		speakerId = token[2]
		segmentStart = float(str(token[3]))
		segmentEnd = float(str(token[4]))
		if previousRecording != self.getAudioFileName():
		    previousRecording = self.getAudioFileName()
		    self.corpusParser.recording(self.getRecordingName(), self.getAudioFileName() )
		spkId = self.corpusParser.speaker( { 'name': speakerId } )
		condId = self.corpusParser.condition( { 'name': speakerId } )
		self.corpusParser.segment(segmentStart, segmentEnd, '[???]')

	print pemFileName, '-->'

class T2PFileParser(TextRowParser):
    # parse
    def parse(self, t2PFileName):
	file = uopen(t2PFileName, self.corpusParser.encoding, 'r')
	lineList = file.readlines()
	uclose(file)

	previousRecording = None
	for line in lineList:
	    token = line.strip().split()
	    #last token contains description
	    description = token[-1].split(',')
	    speakerId = description[8].lower() + "_" + description[9].lower()
	    self.audioFileName_ = description[7]+"_"+speakerId
	    self.recordingName_ = description[7]+"_"+speakerId
	    speakerGender = description[9].lower()
	    conditionId = description[-1].replace(")","").lower()
	    subsetIds = description[0].replace("(","").lower()
	    segmentStart = float(str(description[1]).replace("_",""))
	    segmentEnd = float(str(description[2].replace("_","")))
	    orth = ' '.join( token[0:-1])

	    if previousRecording != self.getAudioFileName():
		previousRecording = self.getAudioFileName()
		self.corpusParser.recording(self.getRecordingName(), self.getAudioFileName() )
	    spkId = self.corpusParser.speaker( { 'name': speakerId , 'gender': speakerGender} )
	    condId = self.corpusParser.condition( { 'name': conditionId } )
	    self.corpusParser.segment(segmentStart, segmentEnd, self.converter.getOrthography(orth))

	print t2PFileName, '-->'


class PemToBlissCorpus:
    """
    special class to convert a single pem file to a single bliss corpus file
    """
    def __init__(self, encoding, trgtFile, isExtractGender):
	self.encoding = encoding
	self.trgtFile = trgtFile
	if isExtractGender:
	    self.spkList =[
		( ('name',   'dummy'),
		  ('gender', 'unknown')
		  ),
		( ('name',   'female dummy'),
		  ('gender', 'female')
		  ),
		( ('name',   'male dummy'),
		  ('gender', 'male')
		  )
		]
	    self.spkNameList = ['unknown', 'female', 'male']
	    self.getSpeakerId = self.getGenderDependentSpeakerId
	else:
	    self.spkList =[
		( ('name',   'dummy'),
		  ('gender', 'unknown')
		  )
		]
	    self.spkNameList = ['unknown']
	    self.getSpeakerId = self.getUnknownSpeakerId
	self.recList = []
	self.condList = []
	self.condLookUp = {}
	self.audioDirectory = None
	self.audioExtension = '.sph'

    def setAudioDirectory(self, audioDirectory):
	if audioDirectory.endswith('/'):
	    self.audioDirectory = audioDirectory[:-1]
	else:
	    self.audioDirectory = audioDirectory
    def getAudioDirectory(self):
	return self.audioDirectory

    def getAudioExtension(self):
	return self.audioExtension
    def setAudioExtension(self, audioExtension):
	if audioExtension[0] != '.':
	    self.audioExtension = '.' + audioExtension
	else:
	    self.audioExtension = audioExtension

    def getAudioFile(self, name):
	if self.audioDirectory is None:
	    audioDir = os.path.dirname(self.srcFile)
	    if not audiDir:
		audioDir = '.'
	else:
	    audioDir = self.audioDirectory
	return audioDir + '/' + name + self.audioExtension

    def getUnknownSpeakerId(self, props):
	return 0

    def getGenderDependentSpeakerId(self, props):
	if props[0] == 'F':
	    return 1
	elif props[0] == 'M':
	    return 2
	else:
	    return 0

    def getConditionId(self, channel, bandwidth, props):
	cond = (
	    ('channel',   channel),
	    ('bandwidth', bandwidth)
	    )
	condId = self.condLookUp.get(cond)
	if condId is None:
	    condId = len(self.condList)
	    self.condList.append(cond)
	    self.condLookUp[cond] = condId
	return condId

    def parse(self, file):
	self.srcFile = file
	fd = uopen(file, self.encoding, 'r')
	self.recList = []
	segList = None
	currentRecName = None
	currentSeg = None
	for row in fd:
	   row = row.strip()
	   if row and not row.startswith(';'):
	       recName, channel, props, start, end, bandwidth = row.split()[:6]
	       if recName != currentRecName:
		   segList = []
		   self.recList.append( Recording(recName, self.getAudioFile(recName), segList) )
		   currentRecName = recName
	       spkId  = self.getSpeakerId(props)
	       condId = self.getConditionId(channel, bandwidth, props)
	       segList.append( Segment(spkId, condId, start, end, '[DUMMY]') )
	uclose(fd)
	self.writeXml()

    def writeXml(self):
	xml = openXml(self.trgFile, self.encoding)
	xml.open('corpus')
	xml.openComment()
	xml.cdata('generated by ' + os.path.abspath(sys.argv[0]))
	# xml.cdata(revision)
	xml.closeComment()
	for id, content in enumerate(self.spkList):
	    xml.open('speaker-description', name=self.spkNameList[id])
	    for tag, cdata in content:
		xml.element(tag, cdata)
	    xml.close('speaker-description')
	for id, content in enumerate(self.condList):
	    xml.open('condition-description', name=str(id))
	    for tag, cdata in content:
		xml.element(tag, cdata)
	    xml.close('condition-description')
	for rec in self.recList:
	    xml.open('recording', name=rec.name, audio=rec.audioFile)
	    for seg in rec.segList:
		xml.open('segment', start=seg.start, end=seg.end)
		xml.empty('speaker', name=self.spkNameList[seg.spkId])
		xml.empty('condition', name=str(seg.condId))
		if seg.orth:
		    xml.element('orth', seg.orth)
		xml.close('segment')
	    xml.close('recording')
	xml.close('corpus')
	closeXml(xml)
	print self.srcFile, '-->', self.trgtFile


# ******************************************************************************
def pem2bliss(file, encoding = 'ascii', target = '-', audioDir = None, audioExt = None, isExtractGender = False):
    writer = PemToBlissCorpus(target, encoding, isExtractGender)
    if audioDir is not None:
	writer.setAudioDirectory(audioDir)
    if audioExt is not None:
	writer.setAudioExtension(audioExt)
    writer.parse(file)

def bliss2bliss(fileList, encoding = 'ascii', target = '-'):
    writer = BlissOrthographyConverterRewriter(encoding, IdConverter(target))
    writer.parse(fileList)

def bliss2trl(fileList, encoding = 'ascii', target = '-'):
    writer = BlissOrthographyConverterRewriter(encoding, TrlConverter(target))
    writer.parse(fileList)

def bliss2old(fileList, encoding = 'ascii', target = '-'):
    writer = BlissOrthographyConverterRewriter(encoding, OldConverter(target))
    writer.parse(fileList)

def bliss2sri(fileList, encoding = 'ascii', target = '-', isSplitAtSentence = False, phraseDelimiter = ' '):
    conv = SriConverter(target)
    conv.setSplitAtSentence(isSplitAtSentence)
    conv.setPhraseDelimiter(phraseDelimiter)
    writer = BlissOrthographyConverterWriter(encoding, conv, not os.path.isdir(target))
    writer.parse(fileList)

def bliss2vocab(fileList, encoding = 'ascii', target = '-', isPhrases = True, phraseDelimiter = ' '):
    if target == '-':
	target = '.'
    writer = WordListExtractor(encoding, target)
    writer.setPhrases(isPhrases)
    writer.setPhraseDelimiter(phraseDelimiter)
    writer.parse(fileList)

def bliss2stm(fileList, encoding = 'ascii', target = '-'):
    writer = BlissOrthographyConverterStmWriter(encoding, StmConverter(target))
    writer.parseSpeakerAndCondition(False)
    writer.parse(fileList)
