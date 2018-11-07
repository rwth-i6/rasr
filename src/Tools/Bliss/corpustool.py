#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import sys
from optparse import OptionParser
from blissCorpusLib import \
     pem2bliss, \
     bliss2bliss, bliss2trl, bliss2old, bliss2sri, bliss2vocab, bliss2stm, \
     CorpusParser, StmFileParser, PemFileParser, T2PFileParser

optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] <corpus-file>+\n' \
    '       processes corpus-file and recursively contained subcorpora\n' \
    '\n' \
    '       List of supported formats\n' \
    '       from:\n' \
    '           bliss (default)\n' \
    '           pem   partition file\n' \
    '           stm   reference file for nist scoring\n' \
    '           t2p   T2P format (HUB4)\n' \
    '       to:\n' \
    '           bliss (default)\n' \
    '           sri   sri language model source data\n' \
    '           vocab extracts a word list and a word-count list,\n' \
    '                 in addition word lists for certain word classes are created\n' \
    '           stm   reference file for nist scoring\n' \
    '           old   still bliss corpus, but orthography ready for old system\n' \
    '           trl   still bliss corpus, but orthography in (rwth-)trl format' \
    )
optparser.add_option("-f", "--from", dest="fromFormat", default="bliss",
		     help= "format to create bliss corpus from", metavar="FORMAT")
optparser.add_option("-t", "--to", dest="toFormat", default="bliss",
		     help= "format to convert bliss corpus to", metavar="FORMAT")
optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
		     help="encoding used for reading plain files and writing files; default is 'ascii'", metavar="ENCODING")
optparser.add_option("-I", "--info", dest="info", action="store_true", default=False,
		     help= "print corpus information")
optparser.add_option('', "--corpus-name", dest="corpusName", default="RWTH",
		     help= "name for the bliss corpus", metavar="STRING")
optparser.add_option('', "--audio-directory", dest="audioDir", default="",
		     help= "directory containing the audio files", metavar="DIRECTORY")
optparser.add_option('', "--audio-extension", dest="audioExt", default=".sph",
		     help= "extension of audio files", metavar="STRING")
optparser.add_option('', "--extract-gender", dest="extractGender", action="store_true", default=False,
		     help= "try to extract gender information (sometimes given)")
optparser.add_option('', "--sri:split-at-punctation", dest="sriSplitAtSentence", action="store_true", default=False,
		     help= "sri: sentence end is only assumed at a punctation mark; not automatically at the end of a segment")
optparser.add_option('', "--sri:phrase-delimiter", dest="sriPhraseDelimiter", default=' ',
		     help= "sri: string inserted between parts of a phrase; default is space", metavar="STRING")
optparser.add_option('', "--vocab:no-phrases", dest="vocabNoPhrases", action="store_true", default=False,
		     help= "vocab: do not maintain phrases")
optparser.add_option('', "--vocab:phrase-delimiter", dest="vocabPhraseDelimiter", default=' ',
		     help= "vocab: string inserted between parts of a phrase; default is space", metavar="STRING")

optparser.add_option("-o", "--output", dest="target", default="-",
		     help="output file or directory (depends on format)", metavar="TARGET")
if len(sys.argv) == 1:
    sys.argv.append('--help')
options, args = optparser.parse_args()

fromFormat = options.fromFormat
toFormat   = options.toFormat
fromSource = toSource  = args
fromTarget = toTarget  = options.target

# if source is not bliss and target is not bliss,
# then  create an intermediate bliss version
if fromFormat != 'bliss':
    if toFormat == 'bliss':
	toFormat = None
    else:
	fromTarget += '.corpus.tmp'
	toSource = [ fromTarget ]
    if fromFormat in ['pem', 'stm', 't2p']:
	corpusParser = CorpusParser(options.corpusName, options.encoding, './xml/')
	if fromFormat == 'pem':
	    corpusParser.setFileParser(PemFileParser(options.audioDir, options.audioExt, options.extractGender))
	elif fromFormat == 'stm':
	    corpusParser.setFileParser(StmFileParser(options.audioDir, options.audioExt, options.extractGender))
	elif fromFormat == 't2p':
	    corpusParser.setFileParser(T2PFileParser(options.audioDir, options.audioExt, options.extractGender))

	for source in fromSource:
	    corpusParser.parse(source)

	if options.info == True:
	    corpusParser.info()

	if toFormat == None:
	    if options.extractGender:
		corpusParser.dumpCorpusBySpeaker(toTarget)
	    else:
		corpusParser.dumpSingleFile(toTarget, options.extractGender)

if toFormat:
    if toFormat == 'bliss':
	bliss2bliss(toSource, options.encoding, toTarget)
    elif toFormat == 'trl':
	bliss2trl(toSource, options.encoding, toTarget)
    elif toFormat == 'old':
	bliss2old(toSource, options.encoding, toTarget)
    elif toFormat == 'sri':
	bliss2sri(toSource, options.encoding, toTarget, options.sriSplitAtSentence, options.sriPhraseDelimiter)
    elif toFormat == 'vocab':
	bliss2vocab(toSource, options.encoding, toTarget, not options.vocabNoPhrases, options.vocabPhraseDelimiter)
    elif toFormat == 'stm':
	bliss2stm(toSource, options.encoding, toTarget)
