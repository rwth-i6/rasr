#!/usr/bin/env python

"""
Scrutinizer

A utility for cleaning word lists.

"""

__version__   = '$Revision$'
__date__      = '$Date$'


# ===========================================================================
# taken from LC-Star WP3: find-funny-characters

# charset
class CharSet:
    def __init__(self, filename=None):
	self.charset = set()
	if filename:
	    self.read(filename)

    def read(self, filename):
	"""
	Character set files are UTF-8 encoded, with three columns
	separated by tabulartor: Unicode codepoint in hexadecimal,
	character, English description
	"""
	for line in uopen(filename, 'UTF-8'):
	    fields = line.split('\t')
	    codepoint, char, description = fields
	    assert codepoint.startswith('U+')
	    codepoint = int(codepoint[2:], 16)
	    assert ord(char) == codepoint
	    self.charset.add(char)

class CharacterTest:
    def __init__(self, charset):
	self.charset = charset
	self.counts = {}

    def __call__(self, word, count):
	for char in word:
	    self.counts[char] = self.counts.get(char, 0) + count
	    if char not in self.charset:
		print (char, word, count)

    def report(self):
	for char in sorted(self.counts):
	    print 'U+%04x\t%s\t%d' % (
		ord(char),
		char.encode('UTF-8'),
		self.counts.get(char, 0)),
	    if char in self.charset:
		print
	    else:
		print '\t!!!'

# ===========================================================================
# idea taken from LC-Star WP3: find-strange-words
# code taken from LC-Star WP3: phon/trust.py

import math, sys
sys.path.append('/u/bisani/sr/src/sequitur')
import mGramCounts, LanguageModel

class PerplexityCalculator(object):
    def __init__(self, order=8):
	self.order = order
	self.symbols = mGramCounts.OpenVocabulary()

    def estimateFromExamples(self, examples):
	"examples is a list of words."
	examples = [ ['<s>'] + list(seq) + ['</s>'] for seq in examples ]
	examples = [ map(self.symbols.index, seq) for seq in examples ]
	counts = mGramCounts.countsFromSequences(examples, self.order)
	self.estimate(counts, len(examples))

    def estimateFromExamplesWithCounts(self, examples):
	"examples is a sequence of (word, count) pairs."
	examples = [ (['<s>'] + list(seq) + ['</s>'], count) for seq, count in examples ]
	examples = [ (map(self.symbols.index, seq),   count) for seq, count in examples ]
	counts = mGramCounts.countsFromSequencesWithCounts(examples, self.order)
	self.estimate(counts, sum(count for seq, count in examples))

    def estimate(self, counts, nExamples):
	counts = list(counts.iter(consolidated=True, sorted=True))
	assert counts[0] == (((), self.symbols.index('<s>')), nExamples)
	del counts[0]
	lmb = LanguageModel.LanguageModelBuilder()
	lmb.setDiscountTypes([LanguageModel.AbsoluteDiscounting])
	lmb.setCountCutoffs([1])
	lmb.setLogFile(sys.stderr)
	self.model = lmb.make(self.symbols, counts, self.order)

    def __call__(self, seq):
	seq = ['<s>'] + list(seq) + ['</s>']
	seq = map(self.symbols.index, seq)
	counts = mGramCounts.countsFromSequence(seq, self.order)

	minProb = 1.0
	totalEntropy = 0.0
	totalEvents = 0
	for (history, predicted), value in counts:
	    p = self.model(history, predicted)
	    minProb = min(minProb, p)
	    totalEntropy += value * math.log(p)
	    totalEvents  += value
	perplexity = math.exp( - totalEntropy / totalEvents)
	return perplexity, 1/minProb


from miscLib import uopen
import os, stat
import cPickle as pickle

def findStrangeWords(filename, encoding):
    """
    Generate a letter N-gram model, and determine the words with the
    highest perplexities according to this model.
    """

    picName = os.path.splitext(os.path.basename(filename))[0] + '.pic'
    if os.path.isfile(picName) and os.stat(picName)[stat.ST_MTIME] >= os.stat(filename)[stat.ST_MTIME]:
	f = open(picName, 'rb')
	orthographicPerplexity = pickle.load(f)
    else:
	orths = []
	for line in uopen(filename, encoding):
	    orths.append(line.strip())
	orthographicPerplexity = PerplexityCalculator()
	orthographicPerplexity.estimateFromExamples(orths)

	f = open(picName, 'wb')
	pickle.dump(orthographicPerplexity, f, pickle.HIGHEST_PROTOCOL)
	f.close()

    return orthographicPerplexity


# ===========================================================================
# find redundant words



# ===========================================================================
from miscLib import uopen

def main(options, args):
    def words():
	for fname in args:
	    for line in uopen(fname, options.encoding):
		if line.startswith('#'): continue
		rank, count, cov1, cov2, word = line.split()
		count = int(count)
		yield word, count

    tests = []

    if options.charset:
	charset = CharSet(options.charset)
	tests.append(CharacterTest(charset.charset))

    if options.auto_examples:
	autoStrangeness = PerplexityCalculator()
	autoStrangeness.estimateFromExamplesWithCounts(words())
	tests.append(autoStrangeness)

    if options.examples:
	modelStrangeness = findStrangeWords(options)
	tests.append(modelStrangeness)

    out = uopen('-', options.encoding, 'w')
    for word, count in words():
	modelPpl, modelMinProb = modelStrangeness(word)
	autoPpl,  autoMinProb  = autoStrangeness(word)
	print >> out, '%s\t%f\t%f\t%f\t%f\t%s' % (count, modelPpl, modelMinProb, autoPpl, autoMinProb, word)


    for word, count in words():
	for test in tests:
	    test(word, count)



if __name__ == '__main__':
    import optparse, sys
    optparser = optparse.OptionParser(
	usage   = '%prog [OPTION]... FILE...\n' + __doc__,
	version = '%prog ' + __version__)
    optparser.add_option(
	'-e', '--encoding', default='UTF-8',
	help='use character set encoding ENC', metavar='ENC')
    optparser.add_option(
	'-c', '--charset',
	help='read character set from FILE', metavar='FILE')
    optparser.add_option(
	'-x', '--examples',
	help='read exmample words from FILE', metavar='FILE')
    optparser.add_option(
	'-a', '--auto-examples', action='store_true',
	help='use examined word list itself as examples')
    options, args = optparser.parse_args()
    main(options, args)
