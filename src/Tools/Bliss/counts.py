#!/usr/bin/env python

"""
Counting words.
"""

__version__   = '$Revision$'
__date__      = '$Date$'

import sys

class CoverageInfo(object):
    def __init__(self, counts, vocabulary=None):
	self.ranking = [ (count, token) for token, count in counts ]
	self.ranking.sort()
	self.ranking.reverse()
	self.nTokens = sum(count for count, token in self.ranking)
	self.singletons = [token for count, token in self.ranking if count == 1]
	self.nSingletons = len(self.singletons)
	self.vocabulary = vocabulary

    def __iter__(self):
	partial = 0
	if self.vocabulary:
	    partialInVoc = 0
	    for rank, (count, token) in enumerate(self.ranking):
		partial += count
		if token in self.vocabulary:
		    partialInVoc += count
		yield (
		    rank,
		    count,
		    partial,
		    token in self.vocabulary,
		    partialInVoc,
		    token)
	else:
	    for rank, (count, token) in enumerate(self.ranking):
		partial += count
		yield (
		    rank,
		    count,
		    partial,
		    token)


class CountsBase(object):
    def add(self, token, count = 1):
	self[token] += count

    def addCounts(self, other):
	for token, count in other.iteritems():
	    self.add(token, count)

    def printStat(self, f = sys.stdout):
	nTokens = reduce(lambda s, cw: s+cw[1], self.iteritems(), 0)
	nSingletons = reduce(lambda s, cw: (cw[1] == 1) and s+1 or s, self.iteritems(), 0)
	print >> f, 'distinct tokens:', len(self)
	print >> f, 'running tokens: ', nTokens
	print >> f, 'singletons:     ', nSingletons

    def exportText(self, f):
	for i in self.counts.iteritems():
	    f.write(repr(i) + '\n')

    def selfCoverage(self, vocabulary = None):
	return CoverageInfo(self.counts.items(), vocabulary)

    def reportCoverage(self, f, vocabulary=None):
	cov = self.selfCoverage(vocabulary)
	if vocabulary:
	    f.write('# <rank>\t<count>\t<in-voc>\t<coverage>\t<coverage w/o singletons>\t<voc. coverage>\t<voc. coverage w/o singletons>\ttoken\n')
	    for rank, count, partial, inVoc, partialInVoc, token in cov:
		try:
		    f.write(u'%d\t%d\t%s\t%f\t%f\t%f\t%f\t%s\n' % (
			rank + 1,
			count,
			{ True: '*', False: '-' }[inVoc],
			float(partial) / float(cov.nTokens),				   # coverage
			float(partial) / float(cov.nTokens - cov.nSingletons), # coverage w/o singletons
			float(partialInVoc) / float(cov.nTokens),					# coverage of vocabulary words
			float(partialInVoc) / float(cov.nTokens - cov.nSingletons), # coverage of vocabulary words w/o singletons
			token))
		except UnicodeEncodeError:
		    print >> sys.stderr, 'Character encoding error for word %s.' % repr(token)
	else:
	    f.write('# <rank>\t<count>\t<coverage>\t<coverage w/o singletons>\ttoken\n')
	    for rank, count, partial, token in cov:
		try:
		    f.write(u'%d\t%d\t%f\t%f\t%s\n' % (
			rank + 1,
			count,
			float(partial) / float(cov.nTokens),				   # coverage
			float(partial) / float(cov.nTokens - cov.nSingletons), # coverage w/o singletons
			token))
		except UnicodeEncodeError:
		    print >> sys.stderr, 'Character encoding error for word %s.' % repr(token)


class InternalCounts(CountsBase):
    def __init__(self):
	self.counts = {}

    def __getitem__(self, token):
	return self.counts.get(token, 0)

    def __setitem__(self, token, count):
	self.counts[token] = count

    def __len__(self):
	return len(self.counts)

    def total(self):
	return sum(self.counts.itervalues())

    def __iter__(self):
	return self.counts.iterkeys()

    def items(self):
	return self.counts.items()

    def iteritems(self):
	return self.counts.iteritems()

    def importText(self, file):
	for line in file:
	    token, count = eval(line)
	    self[token] = count

    def loadCoverage(self, f, limit=None):
	nLines = 0
	for line in f:
	    if limit and nLines >= line: break
	    if line.startswith('#'): continue
	    fields = line.split()
	    self.counts[fields[4]] = int(fields[1])
	    nLines += 1

    def cutoff(self, threshold):
	oldCounts = self.counts
	self.counts = {}
	for token, count in oldCounts.iteritems():
	    if count >= threshold:
		self.counts[token] = count

    def filter(self, predicate):
	oldCounts = self.counts
	self.counts = {}
	for token, count in oldCounts.iteritems():
	    if predicate(token):
		self.counts[token] = count

def countWords(lines):
    counts = InternalCounts()
    for line in lines:
	words = line.split()
	for word in words:
	    counts.add(word)
    return counts

# ===========================================================================
import marshal

class ExternalCounts(CountsBase):
    """
    Represent word counts stored in external database like it was
    standard dictionary.  Uses a Python dictionary as write-through
    cache.
    """

    def __init__(self, filename = None, mutable = False):
	if filename:
	    self.attach(filename, mutable)

    cacheDepth = 10
    cacheLimit = 100000

    def _initCache(self):
	self.cache = [{}]

    def _forget(self):
	self.cache.insert(0, {})
	self.cache = self.cache[:self.cacheDepth]

    def attach(self, filename, mutable = False):
	import bsddb
	self.mutable = mutable
	mode = mutable and 'c' or 'r'
	self.db = bsddb.btopen(filename, mode)
	self._initCache()

    def create(self, filename):
	self.attach(filename, mutable = True)

    def __getitem__(self, token):
	for c in self.cache:
	    if token in c:
		count = c[token]
		break
	else:
	    key = marshal.dumps(token)
	    if key in self.db:
		count = int(self.db[key])
	    else:
		count = 0

	if len(self.cache[0]) >= self.cacheLimit:
	    self._forget()

	self.cache[0][token] = count
	return count

    def _store(self, token, count):
	key = marshal.dumps(token)
	self.db[key] = str(count)

    def __setitem__(self, token, count):
	assert self.mutable
	self._store(token, count)
	if len(self.cache[0]) >= self.cacheLimit:
	    self._forget()
	self.cache[0][token] = count

    def __len__(self):
	return len(self.db)

    def iteritems(self):
	self.db.first()
	for i in xrange(len(self.db)):
	    key, value = self.db.next()
	    token = marshal.loads(key)
	    count = int(value)
	    yield token, count

    def items(self):
	return [ item for item in self.iteritems() ]

    def export(self, file):
	for token, count in self.iteritems():
	    f.write(repr((token, count)) + '\n')

    def importText(self, file, limit=None):
	n = 0
	for line in file:
	    token, count = eval(line)
	    self._store(token, count)
	    n += 1
	    if not n % 10000:
		print n
		print self.db.db.stat()
	    if limit is not None and n >= limit: break
	self._initCache()

# ===========================================================================
def selectWordsByCoverage(counts, targetCoverage):
    coverage = counts.selfCoverage()
    cutoffCount = None
    targetSum = targetCoverage * (coverage.nTokens - coverage.nSingletons)
    for rank, count, partial, token in coverage:
	if partial >= targetSum:
	    cutoffCount = count
	    break
    else:
	raise AssertionError
    print 'cutoff count:', cutoffCount
    result = []
    for rank, count, partial, token in coverage:
	if count >= cutoffCount:
	    result.append(token)
	else:
	    break
    print 'selected words:', len(result)
    return set(result)

# ===========================================================================
from ioLib import zopen

def load(filename, limit=None):
    if filename.endswith('.bt3'):
	assert limit is None
	return ExternalCounts(filename)
    counts = InternalCounts()
    counts.importText(zopen(filename))
    return counts

# ===========================================================================
from ioLib import uopen
import sys

def main(options, args):
    if options.external:
	counts = ExternalCounts(options.external, True)
    else:
	counts = None

    if options.importFiles:
	for filename in options.importFiles:
	    c = InternalCounts()
	    c.importText(uopen(filename))
	    if counts is None:
		counts = c
	    else:
		counts.addCounts(c)

    if counts is None:
	counts = InternalCounts()

    for filename in args:
	for line in uopen(filename, options.encoding):
	    words = line.split()
	    for word in words:
		counts.add(word)

    counts.printStat(sys.stdout)

    if options.coverage:
	if options.vocabulary:
	    vocabulary = set(
		line.strip()
		for line in uopen(options.vocabulary, options.encoding))
	else:
	    vocabulary = None
	counts.reportCoverage(
	    uopen(options.coverage, options.encoding, mode='w'),
	    vocabulary)
    if options.out:
	counts.exportText(uopen(options.out, mode='w'))


if __name__ == '__main__':
    import optparse
    optparser = optparse.OptionParser(
	usage   = '%prog [OPTION]... TEXTFILE...\n' + __doc__,
	version = '%prog ' + __version__)
    optparser.add_option(
	'-e', '--encoding', default='UTF-8',
	help='use character set encoding ENC', metavar='ENC')
    optparser.add_option(
	'-x', '--external',
	metavar='FILE', help='store counts in database FILE')
    optparser.add_option(
	'-i', '--import', action='append', dest='importFiles',
	metavar='FILE', help='read counts from FILE')
    optparser.add_option(
	'-c', '--coverage',
	metavar='FILE', help='write coverage report to FILE')
    optparser.add_option(
	'-v', '--vocabulary',
	metavar='FILE', help='read vocabulary to report selective coverage from FILE')
    optparser.add_option(
	'-o', '--out',
	metavar='FILE', help='write counts to FILE')

    options, args = optparser.parse_args()
    main(options, args)
