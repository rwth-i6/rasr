#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os, sys

gerDir = sys.argv[0]
while os.path.islink(gerDir):
    gerDir = os.path.join(os.path.dirname(gerDir), os.readlink(gerDir))
gerDir = os.path.dirname(gerDir)
binDir = os.path.join(gerDir, 'bin')
libDir = os.path.join(gerDir, 'lib')
cfgDir = os.path.join(gerDir, 'config')
exeDir = os.path.join(gerDir, os.path.join((('..' + os.sep) * 3), 'bin'))
assert os.path.exists(binDir) and os.path.exists(libDir) and os.path.exists(cfgDir)
sys.path = [gerDir, binDir, libDir] + sys.path

from shutil import copyfile, rmtree
from ioLib import uopen, uclose
from miscLib import swapSuffix, mktree
from xmlWriterLib import openXml, closeXml
from stm2blissCorpus import StmToBlissConverter
from htkDir2htkArchive import HtkSlfLatticeArchiver
from htkArchive2blissLexicon import LexiconExtractor
from tranfiltBlissLexicon import TranfiltRules, Hub4NormalizationFilter, TranfiltBlissLexicon
from wer import calculateErrorStatistics


def getDefaultCorpusPath(rootDir, stmFilename):
    return os.path.join(rootDir, swapSuffix(stmFilename, 'corpus.gz', False))

def getDefaultLatticeSuffix(latticeDir):
    return '.lat.gz'

def getDefaultLatticeArchiveDir(rootDir):
    return os.path.join(rootDir, 'archive')

def getDefaultLexiconPath(rootDir, corpusFilename):
    return os.path.join(os.path.join(rootDir, 'lexicon'), swapSuffix(corpusFilename, 'lexicon', True))

def getDefaultFiltLexiconPath(lexiconPath):
    return swapSuffix(lexiconPath, 'filt.lexicon', True)

def getDefaultConfigDir(rootDir):
    return os.path.join(rootDir, 'config')

def getDefaultAlignmentPath(rootDir, corpusFilename):
    return os.path.join(rootDir, swapSuffix(corpusFilename, 'alignment.log', False))

def getDefaultAligner():
    aligner = os.path.abspath(os.path.join(exeDir, 'corpus-aligner'))
    assert os.path.exists(aligner)
    return aligner

def getDefaultResultPath(rootDir, corpusFilename):
    return os.path.join(rootDir, swapSuffix(corpusFilename, 'result', False))


def getValue(value, defaultValue):
    if value:
	return value
    else:
	return defaultValue

def getNormalizedPath(path):
    if path is None or path == "":
	return None
    return os.path.normpath(path)

def valid(trgtPath, *srcPathes):
    if trgtPath is None or trgtPath == "" or trgtPath == ".":
	return False
    elif os.path.exists(trgtPath):
	trgtTime = os.path.getmtime(trgtPath)
	for srcPath in srcPathes:
	    if os.path.getmtime(srcPath) > trgtTime:
		return False
	return True
    else:
	return False

if __name__ == '__main__':
    from optparse import OptionParser
    optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION]\n' \
    '\n' \
    'How to calculate the word and graph error rate for a corpus in NIST\'s STM format\n' \
    'and corresponding lattices in HKT\'s SLF format:\n' \
    'Assume the corpus to be stored in CORPUS_FILE and the lattices in LATTICE_DIR.\n' \
    'For generating a mapping between the rows in CORPUS_FILE and and the\n' \
    'individual lattices, the lattice names have to match the following pattern:\n' \
    'A row, i.e. segment, in an STM file looks like\n' \
    '  "<recording> <channel> <speaker> <start-time> <end-time> ...".\n' \
    'The corresponding lattice must be stored in\n' \
    '  "LATTICE_DIR/<recording>-<start-time>-<end-time>SUFFIX",\n' \
    'where SUFFIX is by default ".lat.gz".\n' \
    'Now, run\n' \
    '   %prog -s CORPUS_FILE -d LATTICE_DIR [-x SUFFIX] [-g GLM_FILE] [-e ENCODING]\n' \
    'GLM_FILE is a file in NIST\'s GLM format and is used to filter\n' \
    'the references and hypothesis before calculating the alignments.\n' \
    'ENCODING specifies the encoding used to generate the lattices.\n' \
    'The calculation of the error rates requires four steps:\n' \
    '1) Translating the STM corpus into RWTH i6 Bliss Corpus Format\n' \
    '2) Extracting a lexicon out of the corpus and the lattices\n' \
    '   Filtering the lexicon using the GLM_FILE, if given\n' \
    '3) Calculating the alignments\n' \
    '4) Calculating the error rates\n' \
    'The following options allows you to specify where to store subresults.\n'
    )
    # bliss corpus
    optparser.add_option("-s", "--stm-file", dest="stmPath", default=None,
			 help="stm file", metavar="FILE")
    optparser.add_option("-c", "--corpus-file", dest="corpusPath", default=None,
			 help="bliss corpus path", metavar="FILE")
    optparser.add_option("", "--corpus-name", dest="corpusName", default="unnamed",
			 help="bliss corpus name; default is 'unnamed'", metavar="STRING")
    # sprint archive
    optparser.add_option("-d", "--lattice-dir", dest="latticeDir", default=None,
			 help="htk slf lattice source directory", metavar="DIR")
    optparser.add_option("-x", "--lattice-suffix", dest="latticeSuffix", default=None,
			 help="htk slf lattice suffix; default is '.lat.gz'", metavar="SUFFIX")
    optparser.add_option("-a", "--lattice-archive-dir", dest="latticeArchiveDir", default="archive",
			 help="htk slf lattice target directory; default is './archive'", metavar="DIR")
    optparser.add_option("", "--copy", dest="copy", action="store_true", default=False,
			 help="do not create symbolic links, but copy lattices")

    # bliss lexicon
    optparser.add_option("-l", "--lexicon-file", dest="lexiconPath", default=None,
			 help="bliss lexicon path", metavar="FILE")

    # filtered bliss lexicon
    optparser.add_option("-g", "--glm-file", dest="glmPath", default=None,
			 help="glm-file to be used to filter the lexicon", metavar="FILE")
    optparser.add_option("", "--filtered-lexicon-file", dest="filtLexiconPath", default=None,
			 help="filtered bliss lexicon path", metavar="FILE")

    # config files
    optparser.add_option("-n", "--config-dir", dest="configDir", default=None,
			 help="config file directory", metavar="FILE")

    # calculate segmentwise leveshtein ditances
    optparser.add_option("-i", "--alignment-file", dest="alignmentPath", default=None,
			 help="file to store alignments", metavar="FILE")
    optparser.add_option("", "--aligner", dest="aligner", default=None,
			 help="executable used for calculating alignments", metavar="EXE")

    # calculate result
    optparser.add_option("-o", "--output", dest="resultPath", default=None,
			 help="write result to FILE", metavar="FILE")

    # general options
    optparser.add_option("", "--case-sensitive", dest="evalCaseSensitive", action="store_true", default=False,
			 help= "do case sensitive error rate calculation")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
			 help="produce verbose output")
    optparser.add_option("-e", "--encoding", dest="encoding", default="iso-8859-1",
			 help="default is 'iso-8859-1'", metavar="ENCODING")
    optparser.add_option("", "--force", dest="force", action="store_true", default=False,
			 help= "force re-creation of files and directories")

    if len(sys.argv) == 1:
	optparser.print_help()
	sys.exit(0)
    options, args = optparser.parse_args()

    stderr = uopen('stderr', 'utf-8', 'w')
    print >> stderr


    # create bliss corpus
    stmPath = getNormalizedPath(options.stmPath)
    if not valid(stmPath):
	corpusPath = getNormalizedPath(options.corpusPath)
	if not valid(corpusPath):
	    print >> stderr, 'Error: Need either a valid stm- or bliss-corpus-file; see --help'
	    sys.exit(1)
	else:
	    print >> stderr, 'Use existing bliss corpus \"' + corpusPath + '\"'
    else:
	corpusPath = getNormalizedPath(getValue(options.corpusPath, getDefaultCorpusPath(os.path.curdir, os.path.basename(stmPath))))
	if options.force or not valid(corpusPath, stmPath):
	    print >> stderr, 'Create bliss corpus from stm-file \"' + stmPath + '\" ...'
	    mktree(os.path.dirname(corpusPath))
	    converter = StmToBlissConverter(options.corpusName)
	    converter.setLog(options.verbose)
	    converter.parse(stmPath, corpusPath, options.encoding)
	    assert os.path.exists(corpusPath)
	    print >> stderr, '\"' + corpusPath + '\" created'
	    del converter
	else:
	    print >> stderr, 'Bliss corpus \"' + corpusPath + '\" is up to date'
    print >> stderr
    corpusFilename = os.path.basename(corpusPath)

    # create sprint archive
    latticeDir = getNormalizedPath(options.latticeDir)
    latticeSuffix = getValue(options.latticeSuffix, getDefaultLatticeSuffix(latticeDir))
    latticeArchiveDir = os.path.normpath(getValue(options.latticeArchiveDir, getDefaultLatticeArchiveDir(os.path.curdir)))
    latticeArchiveConfig = os.path.join(latticeArchiveDir, 'default.config')
    if options.force or not valid(latticeArchiveConfig, corpusPath):
	if os.path.exists(latticeArchiveDir):
	    rmtree(latticeArchiveDir, True)
	mktree(latticeArchiveDir)
	if not valid(latticeDir):
	    print >> stderr, 'Error:', latticeDir, 'is not a valid htk slf lattice source directory; see --help'
	    sys.exit(1)
	archiver = HtkSlfLatticeArchiver(latticeArchiveDir, latticeDir, options.encoding, latticeSuffix, options.copy)
	archiver.setLog(options.verbose)
	print >> stderr, 'Create htk slf archive from htk slf lattices in \"' + latticeDir + '\" ...'
	archiver.parse(corpusPath)
	assert os.path.exists(latticeArchiveConfig)
	print >> stderr, '\"' + latticeArchiveDir + '\" created'
	del archiver
    else:
	print >> stderr, 'Htk slf lattice archive \"' + latticeArchiveDir + '\" is up to date'
    print >> stderr


    # create bliss lexicon
    lexiconPath = getNormalizedPath(getValue(options.lexiconPath, getDefaultLexiconPath(os.path.curdir, corpusFilename)))
    if options.force or not valid(lexiconPath, latticeArchiveConfig):
	print >> stderr, 'Create bliss lexicon from bliss corpus \"' + corpusPath + '\" and htk slf lattice archive \"' + latticeArchiveDir + '\" ...'
	mktree(os.path.dirname(lexiconPath))
	extractor = LexiconExtractor(latticeSuffix, options.encoding)
	extractor.setLog(options.verbose)
	extractor.parse(latticeArchiveDir, corpusPath, lexiconPath)
	assert os.path.exists(lexiconPath)
	print >> stderr, '\"' + lexiconPath + '\" created'
	del extractor
    else:
	print >> stderr, 'Bliss lexicon \"' + lexiconPath + '\" is up to date'
    print >> stderr

    # filter bliss lexicon
    filtLexiconPath = getValue(options.filtLexiconPath, getDefaultFiltLexiconPath(lexiconPath))
    isValid = True
    glmPath = getNormalizedPath(options.glmPath)
    if glmPath is not None:
	if options.force or not valid(filtLexiconPath, glmPath, lexiconPath):
	    isValid = False
	    print >> stderr, 'Create filtered bliss lexicon from bliss lexicon \"' + lexiconPath + '\" and glm-file \"' + glmPath + '\" ...'
	    evalfilt = TranfiltRules()
	    evalfilt.setLog(options.verbose)
	    evalfilt.parse(glmPath, options.encoding)
    else:
	if options.force or not valid(filtLexiconPath, lexiconPath):
	    isValid = False
	    print >> stderr, 'Create filtered bliss lexicon from bliss lexicon \"' + lexiconPath + '\" ...'
	    evalfilt = TranfiltRules()
    if not isValid:
	mktree(os.path.dirname(filtLexiconPath))
	orthfilt = Hub4NormalizationFilter()
	lexFilter = TranfiltBlissLexicon(orthfilt, evalfilt)
	lexFilter.setLog(options.verbose)
	lexFilter.setCapitalize(not options.evalCaseSensitive)
	lexFilter.filter(lexiconPath, filtLexiconPath)
	assert os.path.exists(filtLexiconPath)
	print >> stderr, '\"' + filtLexiconPath + '\" created'
	del orthfilt
	del evalfilt
	del lexFilter
    else:
	print >> stderr, 'Filtered bliss lexicon \"' + filtLexiconPath + '\" is up to date'
    print >> stderr


    # build config files to be used with eval-corpus and eval-lattice
    alignmentPath = getNormalizedPath(getValue(options.alignmentPath, getDefaultAlignmentPath(os.path.curdir, corpusFilename)))
    configDir = getNormalizedPath(getValue(options.configDir, getDefaultConfigDir(os.path.curdir)))
    mktree(configDir)
    defaultConfig = os.path.join(cfgDir, 'default.config')
    evalCorpusConfig = os.path.join(configDir, swapSuffix(corpusFilename, 'alignment.config', False))
    if options.force or not valid(evalCorpusConfig, filtLexiconPath, defaultConfig):
	copyfile(defaultConfig, evalCorpusConfig)
	fd = uopen(evalCorpusConfig, 'utf-8', 'a')
	print >> fd
	print >> fd, '[*.corpus]'
	print >> fd, '%-16s = %s' % ('file', os.path.abspath(corpusPath))
	print >> fd
	print >> fd, '[*.lexicon]'
	print >> fd, '%-16s = %s' % ('file', os.path.abspath(filtLexiconPath))
	print >> fd
	print >> fd, '[*.lattice.archive.reader]'
	print >> fd, '%-16s = %s' % ('path',     os.path.abspath(latticeArchiveDir))
	print >> fd, '%-16s = %s' % ('format',   'htk')
	print >> fd, '%-16s = %s' % ('suffix',   latticeSuffix)
	print >> fd, '%-16s = %s' % ('encoding', options.encoding)
	print >> fd
	print >> fd, '[*]'
	print >> fd, '%-16s = %s' % ('channels.output-channel.file', alignmentPath)
	if options.verbose:
	    print >> fd, '%-16s = %s' % ('warning.channel',  'output-channel,stderr')
	    print >> fd, '%-16s = %s' % ('log.channel',      'output-channel,stdout')
	    print >> fd, '%-16s = %s' % ('progress.channel', 'output-channel,stdout')
	uclose(fd)
	assert os.path.exists(evalCorpusConfig)
	print >> stderr, '\"' + evalCorpusConfig + '\" created'
    else:
	print >> stderr, 'Config file \"' + evalCorpusConfig + '\" is up to date'
    print >> stderr
    defaultEvalLatticeConfig = os.path.join(cfgDir, 'eval-lattice.config')
    evalLatticeConfig = os.path.join(configDir, swapSuffix(corpusFilename, 'config', False))
    if options.force or not valid(evalLatticeConfig, filtLexiconPath, defaultConfig):
	copyfile(defaultConfig, evalLatticeConfig)
	fd = uopen(evalLatticeConfig, 'utf-8', 'a')
	print >> fd
	print >> fd, '[*.lexicon]'
	print >> fd, '%-16s = %s' % ('file', os.path.abspath(filtLexiconPath))
	print >> fd
	print >> fd, '[*.lattice]'
	print >> fd, '%-16s = %s' % ('encoding', options.encoding)
	print >> fd
	print >> fd, '[*]'
	print >> fd, '%-16s = %s' % ('channels.output-channel.file', swapSuffix(corpusFilename, 'log', False))
	print >> fd, '%-16s = %s' % ('log.channel', 'output-channel,stdout')
	if options.verbose:
	    print >> fd, '%-16s = %s' % ('warning.channel',  'output-channel,stderr')
	    print >> fd, '%-16s = %s' % ('progress.channel', 'output-channel,stdout')
	uclose(fd)
	assert os.path.exists(evalLatticeConfig)
	print >> stderr, '\"' + evalLatticeConfig + '\" created'
    else:
	print >> stderr, 'Config file \"' + evalLatticeConfig + '\" is up to date'
    print >> stderr


    # calculate segmentwise levenshtein alignment
    if options.force or not valid(alignmentPath, evalCorpusConfig):
	mktree(os.path.dirname(alignmentPath))
	argv = [
	    '--config=' + evalCorpusConfig
	    ]
	aligner = getValue(options.aligner, getDefaultAligner());
	print >> stderr, 'Calculate alignment for \"' + corpusPath + '\" ...'
##        exitCode = os.spawnvp(os.P_WAIT, aligner, [aligner] + argv)
##        if exitCode != 0:
##            print >> stderr, 'Warning: exit status of \"' + aligner + '\" is', str(exitCode)
	cmd = aligner + ' ' + ' '.join(argv)
	print >> stderr, 'Run', '"' + cmd + '"'
	os.system(cmd)
	assert os.path.exists(alignmentPath)
	print >> stderr, '\"' + alignmentPath + '\" created'
    else:
	print >> stderr, 'Alignment file \"' + alignmentPath + '\" is up to date'
    print >> stderr


    # calculate overall wer
    resultPath = getNormalizedPath(getValue(options.resultPath, getDefaultResultPath(os.path.curdir, corpusFilename)))
    if options.force or not valid(resultPath, alignmentPath):
	print >> stderr, 'Calculate result from \"' + alignmentPath + '\" ...'
	accu, stats = calculateErrorStatistics(alignmentPath, True, True, corpusPath)
	xml = openXml(resultPath, accu.encoding)
	stats.writeXml(xml)
	closeXml(xml)
	assert os.path.exists(resultPath)
	print >> stderr, '\"' + resultPath + '\" created'
	del accu
	del stats
    else:
	print >> stderr, 'Result file \"' + resultPath + '\" is up to date'
    print >> stderr

    uclose(stderr)
