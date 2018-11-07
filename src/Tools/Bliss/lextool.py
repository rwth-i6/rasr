#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
from optparse import OptionParser
from blissLexiconLib import BlissLexicon, PlainLexiconParser


class StackProcessor:
    def __init__(self, stack):
	self.stack = stack
	self.param = []

    def run(self, lex):
	while len(self.stack) > 0:
	    token = self.stack.pop(0)
	    if token in self.cmd:
		cmd = self.cmd[token]
		for i in range(cmd[2]):
		    if len(self.stack) == 0:
			raise Exception('missing operand')
		    self.param.append(self.stack.pop(0))
		if len(self.param) != cmd[1] + cmd[2]:
		    err = 'ERROR: ' + token + ': incorrect number of parameters\n' + ' '.join(self.param)
		    raise Exception(err)
		cmd[0](lex, self.param)
		self.param = []
	    else:
		self.param.append(token)
	lex.closeXmlErr()
	if len(self.param) > 0:
	    err = 'ERROR: remaining parameters\n' + ' '.join(self.param)
	    raise Exception(err)


    def usage(self):
	print "The following commands are supported(#s indicate"
	print "the number and position of required parameters):"
	for cmdName, (cmd, preN, postN, desc) in self.cmd.iteritems():
	    print '%8s %-14s %-8s%s' %  (' '.join(['#' for x in range(preN)]), cmdName, ' '.join(['#' for x in range(postN)]), desc)
	print "use option --help to get more information"

    cmd = {
	'info'          : \
	(lambda lex, t: lex.dumpStatistic('-')    , 0, 0, 'print some useful information'),
	'normal'        : \
	(lambda lex, t: lex.setNormal()           , 0, 0, 'orthography is not modified before added/merged'),
	'lower'          : \
	(lambda lex, t: lex.setLower()            , 0, 0, 'orthography is capitalized before added/merged'),
	'upper'          : \
	(lambda lex, t: lex.setUpper()            , 0, 0, 'orthography is converted to lower case before added/merged'),
	'lower-orth'    : \
	(lambda lex, t: lex.lowerOrthography()    , 0, 0, 'convert orthography to lower case'),
	'upper-orth'    : \
	(lambda lex, t: lex.upperOrthography()    , 0, 0, 'convert orthography to upper case'),
	'add'           : \
	(lambda lex, t: lex.addLexicon(t[0])      , 1, 0, 'add lexicon #; no phonemes are added'),
	'merge'         : \
	(lambda lex, t: lex.mergeLexicon(t[0])    , 1, 0, 'merge lexicon #; phoneme inventories are merged'),
	'add-phrases'   : \
	(lambda lex, t: lex.addCompoundOrthographies(t[0]), 1, 0, 'add phrases from file #; one phrase per line, each part of the phrase has to be in the lexicon'),
	'intersect-orth': \
	(lambda lex, t: lex.intersectOrthFromListFile(t[0]), 1, 0, 'keep only those lemmas such that the orthography is in the word list file #'),
	'intersect-synt': \
	(lambda lex, t: lex.intersectSyntFromListFile(t[0]), 1, 0, 'keep only those lemmas such that the sequence of syntactic tokens is in the word list file #'),
	'remove-orth': \
	(lambda lex, t: lex.removeOrthFromListFile(t[0]), 1, 0, 'keep only those lemmas such that the orthography is not in the word list file #'),
	'remove-orth-special': \
	(lambda lex, t: lex.removeOrthFromListFile(t[0], special=True), 1, 0, 'keep only those special lemmas such that the orthography is not in the word list file #'),
	'remove-synt': \
	(lambda lex, t: lex.removeSyntFromListFile(t[0]), 1, 0, 'keep only those lemmas such that the sequence of syntactic tokens is not in the word list file #'),
	'merge-noise'   : \
	(lambda lex, t: lex.mergeNoiseFile(t[0], False), 1, 0, 'merge noise file #; each row must contain a whitespace-separated list of noise orthographies'),
	'noise-to-si'   : \
	(lambda lex, t: lex.mergeNoiseFile(t[0], True) , 1, 0, 'map noises in file # to silence; for file format see merge-noise'),
	'merge-default' : \
	(lambda lex, t: lex.mergeDefaultLemmas()  , 0, 0, 'merge some default lemmas, e.g. silence'),
	'merge-punct'   : \
	(lambda lex, t: lex.mergeDefaultPunctationMarks(), 0, 0, 'merge default punctation marks, i.e. .?!'),
	'dump-bliss'    : \
	(lambda lex, t: lex.dumpBlissLexicon(t[0]), 0, 1, 'dump current lexicon in bliss format to #'),
	'dump-plain'    : \
	(lambda lex, t: lex.dumpPlainLexicon(t[0]), 0, 1, 'dump current lexicon in plain format to #; see option list'),
	'dump-pron-dict'    : \
	(lambda lex, t: lex.dumpPronDict(t[0]), 0, 1, 'dump current lexicon in pron-dict format to #; see option list'),
	'dump-phon'     : \
	(lambda lex, t: lex.dumpPhonSet(t[0])     , 0, 1, 'dump phoneme inventory to #'),
	'dump-orth'     : \
	(lambda lex, t: lex.dumpOrthSet(t[0])     , 0, 1, 'dump set of orthographic forms to #'),
	'dump-synt'     : \
	(lambda lex, t: lex.dumpSyntTokenSet(t[0]), 0, 1, 'dump set of syntactic token to #'),
	'dump-eval'     : \
	(lambda lex, t: lex.dumpEvalTokenSet(t[0]), 0, 1, 'dump set of evaluation token to #'),
	'dump-stat'     : \
	(lambda lex, t: lex.dumpStatistic(t[0])   , 0, 1, 'same as info, but dumps output to #'),
	'dump-cart'     : \
	(lambda lex, t: lex.dumpCartQuestions(t[0]), 0, 1, 'dump an initial set of cart questions to #'),
	'dump-old'      : \
	(lambda lex, t: lex.dumpOldLexicon(t[0], t[1], t[2], t[3]), 0, 4, 'dump old standard lexicon (made up of four files): <phon> <lex> <orth> <special>'),
	'phoneme-count'      : \
	(lambda lex, t: lex.countPhonemes(t[0], t[1]), 1, 1, 'read file in sri-format and dumps the phoneme count: <sri-file> phoneme-count <output>')
	}


# main
optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <lexicon-file> {command and parameters}+\n'
    '\n' \
    'List of supported formats:\n' \
    '  unknown(default)  either bliss or space-separated\n' \
    '  bliss             our well-known standard format\n' \
    '  space-separated   e.g. hello h e l l o\n' \
    '  tab-seperated     e.g. hello\th e l l o\t[some other information]\n' \
    '  tab-simple        e.g. hello\th e l l o\n' \
    '  callhome          see /u/corpora/lexicon for examples\n' \
    '  grapheme          input is just a word-list file;\n' \
    '                    the phoneme sequence for a word is just the sequence of graphemes\n' \
    '\n' \
    'List of internal options:\n' \
    '  header    the first non-empty, non-comment row in a row based plain format is discarded\n' \
    '  phrases   try to detect and parse phrases')
optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
		     help="encoding used for all written files and all non-xml readed files; default is 'ascii'")
optparser.add_option("-o", "--options", dest="options", default="",
		     help="list of options, seperated by [^a-zA-Z0-9] (see above)")
optparser.add_option("-c", "--check", dest="check", action="store_true", default=False,
		     help="perform consistency check of lexicon on load-time")
optparser.add_option("-u", "--upper", dest="upper", action="store_true", default=False,
		     help="orthography is capitalized before added/merged")
optparser.add_option("-l", "--lower", dest="lower", action="store_true", default=False,
		     help="orthography is converted to lower case before added/merged")
optparser.add_option("-f", "--format", dest="format", default='unknown',
		     help= "format of lexicon to load (see above)")
optparser.add_option("-L", "--list", dest="cmdList", action="store_true", default=False,
		     help="dumps a list of possible commands and required parameters")
optparser.add_option("-v", "--dump-variants", dest="dumpVariants", action="store_true", default=False,
		     help="include pronunciation variants in a plain lexicon dump")
optparser.add_option("-s", "--dump-specials", dest="dumpSpecials", action="store_true", default=False,
		     help="include special lemmas in a plain lexicon dump")
if len(sys.argv) == 1:
    sys.argv.append('--help')
options, args = optparser.parse_args()

if options.cmdList:
    blissLex = BlissLexicon(options.encoding)
    stackProc = StackProcessor(None)
    stackProc.usage()
else:
    blissLex = BlissLexicon(options.encoding)
    if options.lower:
	blissLex.setLower()
    if options.upper:
	blissLex.setUpper()
    blissLex.setDumpVariants(options.dumpVariants)
    blissLex.setDumpSpecials(options.dumpSpecials)
    if options.format == 'unknown':
	if options.check:
	    blissLex.mergeLexicon(args[0], options.options)
	else:
	    blissLex.fastMergeLexicon(args[0], options.options)
    elif options.format == 'bliss':
	if options.check:
	    blissLex.mergeBlissLexicon(args[0])
	else:
	    blissLex.fastMergeBlissLexicon(args[0])
    elif options.format == 'space-separated':
	blissLex.mergePlainLexicon(args[0], PlainLexiconParser.defaultRowFilter, options.options)
    elif options.format == 'tab-separated':
	blissLex.mergePlainLexicon(args[0], PlainLexiconParser.tabSeparatedRowFilter, options.options)
    elif options.format == 'tab-simple':
	blissLex.mergePlainLexicon(args[0], PlainLexiconParser.tabSimpleRowFilter, options.options)
    elif options.format == 'callhome':
	blissLex.mergePlainLexicon(args[0], PlainLexiconParser.callhomeRowFilter, options.options)
    elif options.format == 'grapheme':
	blissLex.mergePlainLexicon(args[0], PlainLexiconParser.orthToGraphemeFilter, options.options)
    else:
	raise Exception('unknown format:' + options.format)
    stackProc = StackProcessor(args[1:])
    stackProc.run(blissLex)
