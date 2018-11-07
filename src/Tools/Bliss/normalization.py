#!/usr/bin/env python
# -*- coding: ISO-8859-1 -*-

"""
Functions for text normalisation.

This module contains several non-language specific functions for
normalizing text, in particular for language modeling purposes.
When used as a tool, it will suggest various mappings based on word
observation counts.

"""

__version__   = '$Revision$'
__date__      = '$Date$'

import re
from miscLib import zopen


# ===========================================================================
def splitSentences(words):
    current = []
    result = [current]
    for w in words:
	current.append(w)
	if w == '</s>':
	    current = []
	    result.append(current)
    if not result[-1]:
	del result[-1]
    return result

def processText(input, processors, output):
    for line in input:
	tokens = line.split()
	for proc in processors:
	    try:
		tokens = proc(tokens)
	    except KeyboardInterrupt:
		raise
	    except:
		import sys, traceback
		print >> sys.stderr, 'processor %s failed on sequence %s' % (repr(proc), repr(tokens))
		traceback.print_exc()
	for sentence in splitSentences(tokens):
	    output.write(' '.join(sentence) + '\n')

# ===========================================================================
# Use this with the string translate() method.
ctt = {
    # spaces
    0x00A0: u' ',     # &nbsp;    non-breaking space
    0x2009: u' ',     # &thinsp;  thin space

    # hyphens
    0x00AD: u'-',     #           soft hyphen
    0x2013: u'-',     # &ndash;   En dash
    0x2014: u'-',     # &mdash;   Em dash
    0x2212: u'-',     # &minus;   Minus sign

    # quotation marks
    0x201c: u'"',     # &ldquo;   left double quotation mark
    0x201d: u'"',     # &rdquo;   right double quotation mark
    0x201e: u'"',     # &bdquo;   double low-9 quotation mark
    0x2018: u'"',     # &lsquo;   left single quotation mark
    0x2019: u'"',     # &rsquo;   right single quotation mark

    # The following is typically mistakenly used instead of &apos;
    0x00B4: u"'",     # &acute;   acute accent

    # other
    0x2022: u'*',     # &bull;    bullet
    0x2026: u'...'}   # &hellip;  Horizontal ellipsis
# discard C0 control characters
ctt.update(dict([ (i, None) for i in range(0x20) ]))
# ... but leave the following untouched
del ctt[0x0009] # HT  \t  horizontal tab
del ctt[0x000A] # LF  \n  line feed
del ctt[0x000D] # CR  \r  carriage return
# discard C1 control characters
ctt.update(dict([ (i, None) for i in range(0x80, 0xA0) ]))

characterTranslationTable = ctt; del ctt

class Sanitizer:
    """
    Some non-ASCII puncuation characters are replaced.
    Very long tokens and control characters are discarded.
    Escape characters not allowed in the preferred encoding
    """

    def __init__(self, encoding=None):
	self.translationTable = characterTranslationTable
	self.ridiculousTokenLength = 100
	self.preferredEncoding = encoding or 'utf-8'

    def containsIllegalCharacters(self, orth):
	try:
	    orth.encode(self.preferredEncoding)
	except UnicodeEncodeError:
	    return True
	return False

    def __call__(self, tokens):
	if not tokens:
	    return tokens
	data = ' '.join(tokens)
	data = data.translate(self.translationTable)
	tokens = data.split()
	tokens = filter(lambda w: len(w) <= self.ridiculousTokenLength, tokens)
	for i in range(len(tokens)):
	    if self.containsIllegalCharacters(tokens[i]):
		tokens[i] = repr(tokens[i])
	return tokens

# ===========================================================================
class PhraseMapper:
    """
    General multi-token mapper

    Mapping is done by looking for left-most longest match.
    """

    def __init__(self, filename=None):
	self.maps = []
	if filename:
	    self.load(zopen(filename))

    def add(self, phrase, replacement):
	phrase      = tuple(phrase)
	replacement = tuple(replacement)
	sl = len(phrase)
	while sl > len(self.maps):
	    self.maps.append(dict())
	self.maps[sl-1][phrase] = replacement

    re_empty_or_comment = re.compile(r'^\s*(#.*)?$')
    re_line = re.compile(r'^(.*)\s+->\s+(.*)$')
    def load(self, f):
	for line in f:
	    if self.re_empty_or_comment.match(line): continue
	    m = self.re_line.match(line)
	    if not m:
		raise ValueError('mapping line must have the form "xxx -> yyy"')
	    phrase      = eval(m.group(1))
	    replacement = eval(m.group(2))
	    if (not (type(phrase)      is tuple  or  type(phrase)      is list) and
		not (type(replacement) is tuple  or  type(replacement) is list)):
		raise ValueError('mapping line item must be a tuple or a list')
	    self.add(phrase, replacement)

    def store(self, filename):
	f = zopen(filename, 'w')
	for mm in self.maps:
	    for phrase, replacement in mm.iteritems():
		print >> f, phrase, ' -> ', replacement

    def size(self):
	return sum(len(mm) for mm in self.maps)

    def __call__(self, tokens):
	result = []
	ii = 0
	while ii < len(tokens):
	    for ll, mm in enumerate(self.maps):
		phrase = tuple(tokens[ii : ii+ll+1])
		if phrase in mm:
		    result += mm[phrase]
		    ii += len(phrase)
		    break
	    else:
		result.append(tokens[ii])
		ii += 1
	return result

    def __mul__(self, other):
	"""
	Composition of mappings
	"""
	result = PhraseMapper()
	first = dict()
	for mm in self.maps:
	    first.update(mm)
	second = dict()
	for mm in other.maps:
	    second.update(mm)
	for phrase, replacement in first.iteritems():
	    result.add(phrase, other(replacement))
	for phrase, replacement in second.iteritems():
	    if phrase not in first:
		result.add(phrase, replacement)
	return result

# ===========================================================================
class AttachedPunctuation:
    """
    Many punctuation marks are typically attached to the words,
    but sometimes punctuation characters are part of the word.

    Quotation marks and brackets are always separate tokens.
    (In Spanish ¿...? and ¡...! bahave like bracket.)

    Other punctuation characters at the end of a word are detached
    unless the form with attached punctuation occurs 10 times more
    frequently than the form without it and at least 5 times in total.
    This should preserve common abbreviations (e.g."d.h.", "usw." ...)
    """

    def __init__(self, counts = None):
	self.rawCounts = counts

    re_unconditionallySplittingPunctuation = re.compile(u'(["«»()\[\]{}¿¡])', re.UNICODE)
    re_tokenWithDetachablePunctuation = re.compile(u'^([-.:,;!?]*)(.+?)([-.:,;!?]*)$', re.UNICODE)

    def splitAtPunctuation(self, rawTokens):
	tokens = []
	for token in rawTokens:
	    tokens += filter(
		lambda s: len(s) > 0,
		self.re_unconditionallySplittingPunctuation.split(token))
	return tokens

    def punctuatedTokenPenalty(self, token):
	return [0, 0.5, 1, 1, 2, 4, 10][min(6, len(token))]

    punctuatedTokenThreshold = 5

    def detachPunctuation(self, rawTokens):
	tokens = []
	for token in rawTokens:
	    m = self.re_tokenWithDetachablePunctuation.match(token)
	    prefix, stem, postfix = m.groups()
	    if not prefix and not postfix:
		tokens.append(token)
		continue
	    best = filter(bool, [prefix, stem, postfix])
	    score = self.rawCounts[stem] * self.punctuatedTokenPenalty(stem)
	    for ll in range(0, len(prefix)+1):
		for rr in range(0, len(postfix)+1):
		    candidate = prefix[ll:] + stem + postfix[:rr]
		    count = self.rawCounts[candidate]
		    if count >= self.punctuatedTokenThreshold and count > score:
			best = filter(bool, [prefix[:ll], candidate, postfix[rr:]])
			score = count
	    tokens += best
	return tokens

    def __call__(self, tokens):
	tokens = self.splitAtPunctuation(tokens)
	if self.rawCounts:
	    tokens = self.detachPunctuation(tokens)
	return tokens


class AllUpperCaseWords:
    """
    In the wild people often write text in all upper-case letters to
    express emphasis.

    Tokens "TOKEN" written entirely in upper case-letters are lower-cased
    if they are four or more characters in length and the lower-case
    version is more frequent.  The most frequent one of the forms "Token",
    "token" or "TOKEN" is chosen.
    """

    def __init__(self, counts):
	self.rawCounts = counts

    def __call__(self, rawTokens):
	tokens = []
	for token in rawTokens:
	    if token.isupper() and len(token) >= 4:
		candidates = [ token, token.lower(), token.capitalize() ]
		ranking = [ (self.rawCounts[c], c) for c in candidates ]
		ranking.sort()
		replacement = ranking[-1][1]
		tokens.append(replacement)
	    else:
		tokens.append(token)
	return tokens


class UsenetEmphasis:
    """
    In the Usenet people use asteriks to put emphsis on words.

    For tokens of the form"*Abc*" the asteriks are detached, iff "Abc" occurs
    more frequently.
    """

    def __init__(self, counts):
	self.rawCounts = counts

    re_emphasizedToken = re.compile('^[*](.*)[*]$', re.UNICODE)
    def __call__(self, rawTokens):
	tokens = []
	for token in rawTokens:
	    m = self.re_emphasizedToken.match(token)
	    if m:
		candidate = m.group(1)
		if self.rawCounts[candidate] > self.rawCounts[token]:
		    tokens += ['*', candidate, '*']
		else:
		    tokens.append(token)
	    else:
		tokens.append(token)
	return tokens


class Compounds:
    """
    Split compound words: Tokens of the form"AB", "AsB", "AesB", "A-B",
    "As-B", "Aes-B" and "A/B" are replaced by "A" "B", iff A and B are
    both six or more characters long, and both A and B occur more frequent
    than the compound form.  (If a hyphen or slash occurs the length limit
    is two characters.)  If the compound form was captialized then "B"
    must be captalized, and "A" may be de-capitalized.  Recursive
    splitting is allowed.
    """

    def __init__(self, counts):
	self.counts = counts
	self.splitMemo = {}

    patterns = [
	# linking token
	# minimum length first component
	# minimum length second component
	# push capitalization
	('-', 3, 3, False),
	('/', 3, 3, False) ]

    germanPatterns = [
	('',    6, 6, True),
	('s',   6, 6, True),
	('es',  6, 6, True),
	('s-',  2, 2, False),
	('es-', 2, 2, False) ]

    def __call__(self, rawTokens):
	tokens = []
	for token in rawTokens:
	    tokens += self.split(token)
	return tokens

    def analyseCompound(self, token):
	min, sum, partition = self.split(token)
	return partition

    def split(self, token):
	tokenCount = self.counts[token]
	candidates = [(tokenCount, tokenCount, (token,))]
	for pattern in self.patterns:
	    candidates += self.splitsForPattern(token, pattern)
	result  = max(candidates)
	return max(result)

    def splitsForPattern(self, token, pattern):
	result = []
	link, minLeftLength, minRightLength, pushCapitalization = pattern
	ll = len(link)
	maxSplitPos = len(token) - minRightLength - ll
	for ii in range(minLeftLength, maxSplitPos+1):
	    if token[ii:ii+ll] == link:
		aa = token[:ii]
		bb = token[ii+ll:]
		result += self.splitCandidates(aa, link, bb, pushCapitalization)
	return result

    def splitCandidates(self, aa, link, bb, pushCapitalization):
	result = []
	aaCount = self.counts[aa]
	bbCount = self.counts[bb]
	if aaCount and bbCount:
	    result.append((
		min(aaCount, bbCount),
		aaCount + bbCount,
		(aa, '_' + link + '_', bb)))
	if pushCapitalization and aa[0].isupper():
	    bb = bb.capitalize()
	    bbCount = self.counts[bb]
	    if aaCount and bbCount:
		result.append((
		    min(aaCount, bbCount),
		    aaCount + bbCount,
		    (aa, '_' + link + '_', bb)))
	return result

# ===========================================================================
class SentenceBoundaries:
    """
    Insert sentence boundary symbols
    """

    sentenceBoundaryTokens = ['.', '?', '!']

    def insertSentenceBoundaryAfterPunctuation(self, inTokens):
	result = []
	for token in inTokens:
	    result.append(token)
	    if token in self.sentenceBoundaryTokens:
		result.append('</s>')
		result.append('<s>')
	return result

    def normalizeSentenceBoundaries(self, inTokens):
	result = []
	state = 1
	for token in inTokens:
	    if token == '<s>' or token == '</s>':
		if state == 0:
		    state = 2
	    else:
		if state >= 2:
		    result.append('</s>')
		if state >= 1:
		    result.append('<s>')
		result.append(token)
		state = 0
	if state != 1:
	    result.append('</s>')
	return result

    def __call__(self, inTokens):
	result = self.insertSentenceBoundaryAfterPunctuation(inTokens)
	result = self.normalizeSentenceBoundaries(result)
	return result

# ===========================================================================
class Punctuation:
    """
    Remove all puncuatuation marks.
    """

#    re_punctuationToken = re.compile(r'^[-_.,:;!¡?¿=+­~&§©®#·*÷×/|¦\\\'"«»()\[\]{}<>]+$', re.UNICODE)
    re_punctuationToken = re.compile(r'^[-_=+­~&§©®#·*÷×/|¦\\\'"«»()\[\]{}<>]+$', re.UNICODE)

    def __call__(self, rawTokens):
	return filter(lambda token: not self.re_punctuationToken.match(token),
		      rawTokens)

# ===========================================================================
class SentenceStartCapitalization:
    """
    Most languages (all?) capitalize the first word of a sentence.
    We would like to change this to the"normal" case of the word,
    i.e. lower-case for common words, upper-case for proper names
    (and nouns in German).

    A captialized word occuring at the beginning of a paragraph or after
    a period (.), a colon (:), a question mark (?) or an exclamation mark (!),
    is decapitalized, unless the capitalized version occurs more frequent
    than the decapitalized one.  (This should preserve capitalization of
    nouns.)

    Idea for improvement: Count how often a word is capitalized when it
    is not in the beginning of sentence.

    context dependent: yes
    prerequisites:     AttachedPunctuation
    """

    def __init__(self, counts):
	self.rawCounts = counts

    re_punctuationToken = Punctuation.re_punctuationToken
    re_capitalizationInducingPunctuationToken = re.compile('^[.:!¡?¿]+$', re.UNICODE)

    def __call__(self, rawTokens):
	tokens = []
	isCapitalizationPoint = True
	for token in rawTokens:
	    if token == "<s>":
		pass
	    elif self.re_capitalizationInducingPunctuationToken.match(token):
		isCapitalizationPoint = True
	    elif self.re_punctuationToken.match(token):
		pass
	    else:
		if isCapitalizationPoint and token[0].isupper() and (not token.isupper() or (len(token) == 1)):
		    candidate = token.lower()
#                    print >> sys.stderr, token, ' upper: ', self.rawCounts[token], 'lower: ', self.rawCounts[candidate], '\n'
		    if ((self.rawCounts[candidate] - self.rawCounts[token]) > 0) or (self.rawCounts[candidate] >= 1000):
			token = candidate
		isCapitalizationPoint = False
	    tokens.append(token)
	return tokens

# ===========================================================================
from counts import InternalCounts
import sys

def makeMapper(counts, mapperClass):
    """
    Given word observation counts and a mapping class, create a
    replacement table.
    """
    result = PhraseMapper()
    newCounts = InternalCounts()
    mapper = mapperClass(counts)
    for word in counts:
	phrase = [word]
	count = counts[word]
	replacement = mapper(phrase)
	if replacement != phrase:
	    result.add(phrase, replacement)
	for replacementToken in replacement:
	    newCounts.add(replacementToken, count)
    return result, newCounts

def makeMapperIterated(counts, mapperClasses):
    """
    Iteratively apply a set of mapping classes until no futher mapping
    are found.
    """
    mapper = PhraseMapper()
    hasChanged = True
    while hasChanged:
	hasChanged = False
	for mapperClass in mapperClasses:
	    print mapperClass.__name__
	    newMapper, newCounts = makeMapper(counts, mapperClass)
	    print 'new mappings:   ', newMapper.size()
	    hasChanged = hasChanged or newMapper.size() > 0
	    mapper *= newMapper
	    counts  = newCounts
	    print 'mappings:       ', mapper.size()
	    counts.printStat(sys.stdout)
	    print
    return mapper, counts

# ===========================================================================
from stepper import Stepper
from ioLib import uopen, zopen
import counts
import os

class TextNormalizer(Stepper):
    encoding = 'utf-8'

    def __init__(self, name, options):
	assert options.target and os.path.isdir(options.target)
	self.target_ = options.target
	Stepper.__init__(
	    self,
	    os.path.join(self.target_, '__dict__'),
	    name,
	    options)

    def readText(self, which):
	fname = os.path.join(self.target_, self.name + '-' + which + '.text.gz')
	return uopen(fname, self.encoding, 'r')

    def writeText(self, which):
	fname = os.path.join(self.target_, self.name + '-' + which + '.text.gz')
	return uopen(fname, self.encoding, 'w')

    def countWords(self, which):
	cnt = counts.countWords(self.readText(which))
	cnt.printStat(sys.stdout)
	fname = os.path.join(self.target_, self.name + '-' + which + '.cov.gz')
	cnt.reportCoverage(uopen(fname, self.encoding, 'w'))
	fname = os.path.join(self.target_, self.name + '-' + which + '.counts.gz')
	cnt.exportText(zopen(fname, 'w'))

    def loadCounts(self, which):
	fname = os.path.join(self.target_, self.name + '-' + which + '.counts.gz')
	return counts.load(fname)

    def makeMapper(self, which, mapperClasses):
	cnt = self.loadCounts(which)
	mapping, cnt = makeMapperIterated(cnt, mapperClasses)
	print 'mappings:', mapping.size()
	fname = os.path.join(self.target_, self.name + '-' + which + '.map')
	mapping.store(fname)

    def loadMapper(self, which):
	return PhraseMapper(os.path.join(self.target_, self.name + '-' + which + '.map'))

# ===========================================================================
from counts import load as loadCounts

mappersByName = {
    'punctuation'      : AttachedPunctuation,
    'upper-case-words' : AllUpperCaseWords,
    'usenet-emphasis'  : UsenetEmphasis,
    'sentence-caps'    : SentenceStartCapitalization
    }

import sys

def main(options, args):
    if len(args) < 2:
	return

    counts = loadCounts(args[0])
    counts.printStat(sys.stdout)
    print

    mapper, newCounts = makeMapperIterated(
	counts,
	[ mappersByName[what] for what in args[1:] ])

    if options.map:
	mapper.store(options.map)
    if options.counts:
	newCounts.exportText(zopen(options.counts, 'w'))


if __name__ == '__main__':
    import optparse, tool
    optparser = optparse.OptionParser(
	usage   = '%prog [OPTION]... FILE...\n' + __doc__,
	version = '%prog ' + __version__)
    optparser.add_option(
	'-e', '--encoding', default='UTF-8',
	help='use character set encoding ENC', metavar='ENC')
    optparser.add_option(
	'-m', '--map',
	help='store mapping rule in FILE', metavar='FILE')
    optparser.add_option(
	'-c', '--counts',
	help='store new (mapped) counts in FILE', metavar='FILE')

    options, args = optparser.parse_args()
    main(options, args)
