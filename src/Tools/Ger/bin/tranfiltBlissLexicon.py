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
from stringLib import find_unmasked, next_in, next_not_in
from xmlWriterLib import openXml, closeXml
from xmlParserLib import SimpleXmlParser


# Filter applying rules from a GLM-file to an eval token
# input:  eval token
# output: list of eval token sequence(s)
class TranfiltRules:
    def __init__(self):
	self.rules_ = {}
	self.isLog = False

    def setLog(self, isLog):
	self.isLog = isLog

    def __call__(self, orth):
	return self.rules_.get(orth, None)

    def appendToken_(self, l, t):
	if t != '@':
	    # split even strings surrounded by squared brackets
	    for u in t.split():
		l.append(u)

    def parseString_(self, s):
	l = []
	b = next_not_in(s, ' \t', 0, len(s))
	self.assertRuleSyntax_(b < len(s), 'missing token')
	while b < len(s):
	    if s[b] == '[':
		e = find_unmasked(s, ']', b+1)
		self.assertRuleSyntax_(e != -1, 'missing closing bracket "]"')
		self.appendToken_(l, s[b+1:e])
		b = next_not_in(s, ' \t', e + 1, len(s))
	    else:
		e = s.find(' ', b+1)
		if e == -1:
		    self.appendToken_(l, s[b:])
		    b = len(s)
		else:
		    self.appendToken_(l, s[b:e])
		    b = next_not_in(s, ' \t', e + 1, len(s))
	return l

    def parseHead_(self, s):
	return self.parseString_(s)

    def emptyContext_(self):
	self.ruleWarning_('Cannot consider empty context, use "[ ] __ [ ]" instead.')

    def parseContext_(self, s):
	self.assertRuleSyntax_(s.find('__') != -1, 'missing wildcard "__" in context')
	l, r = ( t.strip() for t in s.split('__') )
	if not l == r == '[ ]':
	    self.ruleWarning_('Cannot consider context, use "[ ] __ [ ]" instead.')

    def parseTail_(self, s):
	self.assertRuleSyntax_(s, 'tail is empty')
	i = next_not_in(s, ' \t', 0, len(s))
	if s[i] == '[':
	    isBracket = True
	    i = next_not_in(s, ' \t', i+1, len(s))
	else:
	    isBracket = False
	if s[i] == '{':
	    j = find_unmasked(s, '}', i)
	    e = j + 1
	    if isBracket:
		e = next_not_in(s, ' \t', e, len(s))
		self.assertRuleSyntax_(e < len(s) and s[e] == ']', 'missing closing bracket "]"')
		e += 1
	    e = next_not_in(s, ' \t', e, len(s))
	    if e == len(s):
		self.emptyContext_()
	    else:
		self.assertRuleSyntax_(s[e] == '/', 'missing context separator "/"')
		self.parseContext_(s[e+1:])
	    l = []
	    i += 1
	    e = find_unmasked(s, '/', i)
	    while e != -1 and e < j:
		l.append(self.parseString_(s[i:e]))
		i = e + 1
		e = find_unmasked(s, '/', i)
	    l.append(self.parseString_(s[i:j]))
	    return l
	else:
	    i = find_unmasked(s, '/', i)
	    if i == -1:
		evalStr = s
		self.emptyContext_()
	    else:
		evalStr, contextStr = s[:i], s[i+1:]
		self.parseContext_(contextStr)
	    return [ self.parseString_(evalStr) ]

    def parseRule_(self, s):
	self.currentRule_ = s
	parts = s.split('=>')
	self.assertRuleSyntax_(len(parts) == 2, 'missing context separator "/"')
	head, tail = self.parseHead_(parts[0].strip()), self.parseTail_(parts[1].strip())
	if len(head) == 1:
	    head = head[0]
	    evalTokSeqSet = self.rules_.get(head, None)
	    if evalTokSeqSet is None:
		self.rules_[head] = tail
	    else:
		self.ruleWarning_('A rule for head "' + head + '" does already exist; add new rule as alternative.')
		for evalTokSeq in tail:
		    if evalTokSeq not in evalTokSeqSet:
			evalTokSeqSet.append(evalTokSeq)
	else:
	    self.ruleWarning_('Head must consist of a single token; discard rule.')
	del self.currentRule_

    def assertRuleSyntax_(self, b, m):
	if not b:
	    print >> self.err, 'Error in line', str(self.currentLine_) + '/ rule "' + self.currentRule_ + '".'
	    print >> self.err, '    Cannot parse rule:', m
	    sys.exit(1)

    def ruleWarning_(self, m):
	print >> self.err, 'Warning in line', str(self.currentLine_) + '/ rule "' + self.currentRule_ + '".'
	print >> self.err, '    ', m

##* name "en971128.glm"
##* desc "The Universal GLM file for the ARPA Hub4-E and Hub5-E Eval Test Alternate Spellings and Contractions Map"
##* format = 'NIST1'
##* max_nrules = '2500'
##* copy_no_hit = 'T'
##* case_sensitive = 'F'
    def parseHeader_(self, s):
	if s.startswith('*'):
	    pass
	else:
	    self.parseContent_ = self.parseRule_
	    self.parseContent_(s)

    def parseLine_(self, s):
	i = s.find(';;')
	if i != -1:
	    s = s[:i]
	s = s.strip()
	if s:
	    self.parseContent_(s)

    def parse(self, path, encoding):
	self.parseContent_ = self.parseHeader_
	self.err = uopen('stderr', encoding, 'w')
	fd = uopen(path, encoding, 'r')
	self.currentLine_ = 0
	for line in fd:
	    self.currentLine_ += 1
	    self.parseLine_(line)
	del self.currentLine_
	uclose(fd)
	uclose(self.err)

    def write(self, fd):
	for head, tail in self.rules_.iteritems():
	    if len(tail) == 1:
		print >> fd, '"' + head + '"' , '=>', ' '.join( ('"' + tok + '"' for tok in tail[0] ) )
	    else:
		print >> fd, '"' + head + '"' , '=>', '{', ' / '.join( ' '.join( ('"' + tok + '"' for tok in tokSeq) ) for tokSeq in tail ), '}'


# Filter orthographies by a predefined, yet hardwired set of rules
# input:  orthography
# output: list of eval token sequence(s)
class NoNormalizationFilter:
    def __call__(self, o):
	return [ (o,) ]

class Hub4NormalizationFilter:
    def __init__(self):
	import re
	self.reHyphen = re.compile(r'((?<=\S)\-(?=\S))|(\s-\s)|(^-\s)|(\s-$)')

    def __call__(self, o):
	assert o
	# 1) delete words in {} brackets
	if o[0] == '{' and o[-1] == '}':
	    return [ () ]
	# 2) delete words in [] brackets
	if o[0] == '[' and o[-1] == ']':
	    return [ () ]
	evalTokSeqList = []
	# 3) make words in () optinal deletable
	if o[0] == '(' and o[-1] == ')':
	    evalTokSeqList.append( () )
	    o = o[1:-1]
	    assert o
	# 4) split words at _ (=phrases)
	o = o.replace('_', ' ')
	# 5) split words at hyphens, if the hyphen is not a broken word marker
	o = self.reHyphen.sub(' ', o)
	assert o
	evalTokSeqList.append(o.split())
	return evalTokSeqList


# tranfilt lexicon
class Lemma:
    __slots__ = ('attr', 'orthSet', 'phonSeqList', 'syntTokSeq', 'evalTokSeqList')

    def __init__(self):
	self.reset()

    def reset(self):
	self.attr = None;
	self.orthSet = [];
	self.phonSeqList = [];
	self.syntTokSeq = None
	self.evalTokSeqList = []

    def writeXml(self, xml):
	assert self.attr is not None;
	xml.open('lemma', self.attr)
	for orth in self.orthSet:
	    xml.element('orth', orth)
	for phonSeq in self.phonSeqList:
	    xml.element('phon', phonSeq)
	if self.syntTokSeq is not None:
	    xml.open('synt')
	    for tok in self.syntTokSeq:
		xml.element('tok', tok)
	    xml.close()
	for evalTokSeq in self.evalTokSeqList:
	    xml.open('eval')
	    for tok in evalTokSeq:
		xml.element('tok', tok)
	    xml.close()
	xml.close()


class TranfiltBlissLexicon(SimpleXmlParser):
    def __init__(self, orthfilt, evalfilt):
	SimpleXmlParser.__init__(self)
	self.orthfilt = orthfilt
	self.evalfilt = evalfilt
	self.startElement = self.startLexicon
	self.endElement   = self.endLexicon
	self.cdata = ''
	self.tokSeq = []
	self.lemma = Lemma()
	self.isLog = False
	self.isCapitalize = False

    def setLog(self, isLog):
	self.isLog = isLog

    def setCapitalize(self, isCapitalize):
	self.isCapitalize = isCapitalize

    def characters(self, cdata):
	self.cdata += cdata

    def writeCdata(self):
	self.cdata = self.cdata.strip()
	if self.cdata:
	    self.xml.write(self.cdata)
	    self.cdata = ''

    def startLexicon(self, name, attr):
	self.writeCdata()
	if name == 'lemma':
	    self.lemma.attr = attr
	    self.startElement = self.startLemma
	    self.endElement   = self.endLemma
	else:
	    self.xml.open(name, attr)

    def endLexicon(self, name):
	self.writeCdata()
	self.xml.close()

    def startLemma(self, name, attr):
	self.cdata = ''
	if name == 'synt' or name == 'eval':
	    self.startElement = self.startToken
	    self.endElement   = self.endToken

    def endLemma(self, name):
	if name == 'lemma':
	    self.processLemma(self.lemma)
	    self.lemma.reset()
	    self.startElement = self.startLexicon
	    self.endElement   = self.endLexicon
	else:
	    if name == 'orth':
		self.lemma.orthSet.append(self.cdata.strip())
	    elif name == 'phon':
		self.lemma.phonSeqList.append(self.cdata.strip())
	    elif name == 'synt':
		self.lemma.syntTokSeq = self.tokSeq
		self.tokSeq = []
	    elif name == 'eval':
		self.lemma.evalTokSeqList.append(self.tokSeq)
		self.tokSeq = []
	    else:
		print >> self.stderr, 'Error: Unexpected child of lemma "' + name + '"'
		sys.exit(1)
	self.cdata = ''

    def startToken(self, name, attr):
	self.cdata = ''
	if name != 'tok':
	    print >> self.stderr, 'Error: Unexpected element "' + name + '" in token sequence'
	    sys.exit(1)

    def endToken(self, name):
	if name != 'tok':
	    self.startElement = self.startLemma
	    self.endElement   = self.endLemma
	    self.endElement(name)
	else:
	    self.tokSeq.append(self.cdata.strip())
	self.cdata = ''


    def processLemma(self, lemma):
	if not lemma.evalTokSeqList:
	    if lemma.orthSet:
		lemma.evalTokSeqList = self.orthfilt(lemma.orthSet[0])
	if lemma.evalTokSeqList:
	    if self.isCapitalize:
		lemma.evalTokSeqList = [ [ tok.upper() for tok in tokSeq ] for tokSeq in lemma.evalTokSeqList ]
	    lemma.evalTokSeqList = self.filterEvalTokSeqList(lemma.evalTokSeqList)
	lemma.writeXml(self.xml)

    #  et    :          evaluation token
    #  ets   :          evaluation token    sequence
    #  etsl  :          evaluation token    sequence list
    # fetss  : filtered evaluation token subsequence
    # fetssl : filtered evaluation token subsequence list
    # fets   : filtered evaluation token    sequence
    # fetsl  : filtered evaluation token    sequence list
    def filterEvalTokSeqList(self, etsl):
	fetsl = []
	for ets in etsl:
	    if not ets:
		fetsl.append(ets)
	    elif len(ets) == 1:
		tmp_fetsl = self.evalfilt(ets[0])
		if tmp_fetsl is not None:
		    fetsl = tmp_fetsl
		else:
		    fetsl.append(ets)
	    else:
		tmp_fetsl = []
		for et in ets:
		    fetssl = self.evalfilt(et)
		    if fetssl is not None:
			if tmp_fetsl:
			    headl, tmp_fetsl = tmp_fetsl, []
			    for head in headl:
				for fetss in fetssl:
				    tmp_fetsl.append(head + fetss)
			else:
			   tmp_fetsl = [ list(fetss) for fetss in fetssl ]
		    else:
			if tmp_fetsl:
			    for fets in tmp_fetsl:
				fets.append(et)
			else:
			    tmp_fetsl.append([et])
		fetsl.extend(tmp_fetsl)
	return fetsl

    def startFile(self, path, encoding):
	self.xml = openXml(self.outPath, encoding)

    def endFile(self, path):
	closeXml(self.xml)

    def filter(self, fromPath, toPath):
	self.outPath = toPath
	self.parse(fromPath)
	if self.isLog:
	    print >> sys.stderr, fromPath, '-->', toPath


if __name__ == '__main__':
    from optparse import OptionParser
    optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <bliss-lexicon>'
    )
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
			 help="produce verbose output")
    optparser.add_option("-c", "--capitalize", dest="capitalize", action="store_true", default=False,
			 help="capitalize evaluation tokens")
    optparser.add_option("-n", "--normalize", dest="normalize", action="store_true", default=False,
			 help="try to normalize orthographies according to the rules used by the hub4-scorer")
    optparser.add_option("-g", "--glm", dest="glmPath", default="",
			 help="glm-file to be used to filter the lexicon", metavar="FILE")
    optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
			 help="encoding used for reading the glm file; default is 'ascii'", metavar="ENCODING")
    optparser.add_option("-o", "--output", dest="output", default="-",
			 help="write filtered lexicon to FILE; default is stdout", metavar="FILE")

    if len(sys.argv) == 1:
	sys.argv.append('--help')
    options, args = optparser.parse_args()
    evalfilt = TranfiltRules()
    evalfilt.setLog(options.verbose)
    if options.glmPath:
	print >> sys.stderr, 'Parse filter rules from "' + options.glmPath +'" ...'
	evalfilt.parse(options.glmPath, options.encoding)
    if options.normalize:
	orthfilt = Hub4NormalizationFilter()
    else:
	orthfilt = NoNormalizationFilter()
    lexFilter = TranfiltBlissLexicon(orthfilt, evalfilt)
    lexFilter.setCapitalize(options.capitalize)
    lexFilter.setLog(options.verbose)
    print >> sys.stderr, 'Filter lexicon "' + args[0] +'" ...'
    lexFilter.filter(args[0], options.output)
