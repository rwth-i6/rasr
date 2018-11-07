#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os
import sys
from optparse import OptionParser
from os.path import exists
from blissLexiconLib import BlissLexicon, CostaLog, PlainLexiconParser
from miscLib import uopen, uclose

def setG2pEnvironment():
    libPathList = [
	'/u/bisani/lib/python',
	'/u/bisani/sr/g2p/lib'
	]
    binPathList = [
	'/u/bisani/sr/g2p/lib'
	]
    sys.path.extend(libPathList)
    pythonPath = ':'.join(libPathList) + ':' + os.getenv('PYTHONPATH', '.')
    os.putenv('PYTHONPATH', pythonPath)
    path = ':'.join(binPathList) + ':' +  os.getenv('PATH', '.')
    os.putenv('PATH', path)
setG2pEnvironment()
#import g2p
#import tool


class LexiconMaker:
    def __init__(self, options):
	self.costaLog = None
	self.encoding = options.encoding
	self.context = options.context
	self.variants = options.variants
	self.fforce = options.fforce
	self.force = options.force
	self.setFileNameList(options)


    # ################# auxiliary function ##########################
    def exists(self, name):
	if self.fforce:
	    return False
	elif self.force and name != self.modelFile:
	    return False
	else:
	    return exists(name)

    def makePrefix(self, path, prefix):
	if path == None:
	    return None
	i = path.rfind('/')
	if i == -1:
	    directory = ''
	    name = path
	else:
	    directory = path[:i+1]
	    name = path[i+1:]
	return directory + prefix + name

    def makeAppendix(self, path, appendix, isPreserveGzip = False):
	if path == None:
	    return None
	if path[-3:] == '.gz':
	    path = path[:-3]
	    if isPreserveGzip:
		appendix += '.gz'
	i = path.rfind('.')
	if i == -1:
	    return path + '.' + appendix
	else:
	    return path[:i+1] + appendix

    def setCostaLog(self):
	if self.costaLog:
	    return True
	elif self.costaLogFile:
	    self.costaLog = CostaLog(self.costaLogFile)
	    return True
	else:
	    return False

    def setFileNameList(self, options):
	self.costaLogFile        = options.costaLogFile
	self.lexiconFile         = options.lexiconFile
	self.trainLexiconFile    = options.trainLexiconFile
	self.modelFile           = options.modelFile
	self.missingWordFile     = options.missingWordFile
	self.missingTransFile    = options.missingTransFile
	self.completeLexiconFile = options.lexiconOut
	# create missing file names
	if not self.lexiconFile:
	    if self.setCostaLog():
		self.lexiconFile = self.costaLog.getLexiconFile()
	if not self.trainLexiconFile:
	    self.trainLexiconFile = self.lexiconFile
	if not self.modelFile:
	    self.modelFile = self.makeAppendix(self.trainLexiconFile, 'g2p')
	if not self.missingWordFile:
	    self.missingWordFile = self.makeAppendix(self.costaLogFile, 'missing.vocab')
	if not self.missingTransFile:
	    self.missingTransFile = self.makeAppendix(self.modelFile, 'missing.plain.lex')
	if not self.completeLexiconFile:
	    self.completeLexiconFile = self.makeAppendix(self.lexiconFile, 'complete.lexicon', True)


    # ################# g2p model ##########################
    def initG2pModel(self, modelFile):
	cmd = 'g2p.py'\
	      + ' --encoding='    + self.encoding \
	      + ' --train='       + self.trainLexiconFile\
	      + ' --write-model=' + modelFile
	print >> sys.stderr, cmd
	if os.system(cmd) > 0:
	    raise 'ERROR: initializing model failed'

    def rampG2pModel(self, baseFile, modelFile):
	cmd = 'g2p.py'\
	      + ' --encoding='    + self.encoding \
	      + ' --ramp'\
	      + ' --model='       + baseFile\
	      + ' --train='       + self.trainLexiconFile\
	      + ' --write-model=' + modelFile
	print >> sys.stderr, cmd
	if os.system(cmd) > 0:
	    raise 'ERROR: ramping model failed'

    def applyG2pModel(self):
	cmd = 'g2p.py'\
	      + ' --encoding=' + self.encoding \
	      + ' --model='    + self.modelFile\
	      + ' --apply='    + self.missingWordFile
	if self.variants > 0.0:
	    cmd += \
		' --variants=' + str(self.variants)
	cmd += \
	    ' > ' + self.missingTransFile
	print >> sys.stderr, cmd
	if os.system(cmd) > 0:
	    raise 'ERROR: transcription failed'

    def train(self):
	if self.context == 1:
	    print >> sys.stderr, 'train', self.modelFile
	    self.initG2pModel(self.modelFile)
	else:
	    prefix = '_n'
	    modelFileList = []
	    for n in range(1, self.context):
		modelFile = self.makePrefix(self.modelFile, prefix + str(n) + '_')
		modelFileList.append(modelFile)

	    # init
	    modelFile = modelFileList[0]
	    if self.exists(modelFile):
		print >> sys.stderr, 'recover from', modelFile
	    else:
		print >> sys.stderr, 'train', modelFile
		self.initG2pModel(modelFile)
	    # ramp
	    for i in range(1, self.context - 1):
		modelFile = modelFileList[i]
		if self.exists(modelFile):
		    print >> sys.stderr, 'recover from', modelFile
		else:
		    print >> sys.stderr, 'train', modelFile
		    self.rampG2pModel(modelFileList[i-1], modelFile)
	    print >> sys.stderr, 'train', self.modelFile
	    self.rampG2pModel(modelFileList[-1], self.modelFile)
	    # clean up
	    for model in modelFileList:
		print >> sys.stderr, 'remove', model
		os.unlink(model)

    def transcribe(self):
	self.applyG2pModel()

    # ################# make ##########################
    def ensureMissingWordList(self):
	if not self.missingWordFile:
	    raise 'cannot get missing words'
	if self.exists(self.missingWordFile):
	    print >> sys.stderr, 'missing word file does already exist:', self.missingWordFile
	else:
	    if self.setCostaLog():
		print >> sys.stderr, 'get missing words from costa-log-file', self.costaLogFile
		self.missingWords = self.costaLog.getMissingWordList()
		fd = uopen(self.missingWordFile, self.encoding, 'w')
		print >> sys.stderr, 'store missing words:', fd.name
		for word in self.missingWords:
		    fd.write(word + '\n')
		uclose(fd)
	    else:
		raise 'cannot create missing word file', self.missingWordFile

    def ensureG2pModel(self):
	if not self.modelFile:
	    raise 'cannot load or train g2p-model'
	if self.exists(self.modelFile):
	    print >> sys.stderr, 'g2p-model does already exist:', self.modelFile
	    self.model = self.modelFile
	    return
	elif self.trainLexiconFile:
	    print >> sys.stderr, 'train g2p-model using', self.trainLexiconFile
	    self.train()
	else:
	    raise 'cannot train g2p-model', self.modelFile

    def ensureMissingTranscriptionList(self):
	if not self.missingTransFile:
	    raise 'cannot load or estimate missing transcriptions'
	if self.exists(self.missingTransFile):
	    print >> sys.stderr, 'transcription file does already exist:', self.missingTransFile
	else:
	    self.ensureMissingWordList()
	    self.ensureG2pModel()
	    print >> sys.stderr, 'use model', self.modelFile, 'to transcribe missing words in', self.missingWordFile
	    print >> sys.stderr, 'store transcriptions:', self.missingTransFile
	    self.transcribe()

    def ensureCompleteLexicon(self):
	if not self.lexiconFile:
	    raise 'cannot create lexicon, because no lexicon file is specified'
	if self.exists(self.completeLexiconFile):
	    print >> sys.stderr, 'complete lexicon file does already exist:', self.completeLexiconFile
	else:
	    self.ensureMissingTranscriptionList()
	    print >> sys.stderr, 'merge ', self.lexiconFile, 'and', self.missingTransFile, 'into', self.completeLexiconFile
	    blissLexicon = BlissLexicon(self.encoding)
	    blissLexicon.fastMergeBlissLexicon(self.lexiconFile)
	    blissLexicon.addPlainLexicon(self.missingTransFile, PlainLexiconParser.tabSeparatedRowFilter)
	    blissLexicon.dumpBlissLexicon(self.completeLexiconFile)

    def run(self, cmd):
	if 'l' in cmd:
	    print >> sys.stderr, 'make complete lexicon ...'
	    self.ensureCompleteLexicon()
	elif 't' in cmd:
	    print >> sys.stderr, 'make missing transcription ...'
	    self.ensureMissingTranscriptionList()
	else:
	    if 'm' in cmd:
		print >> sys.stderr, 'make g2p-model ...'
		self.ensureG2pModel()
	    if 'w' in cmd:
		print >> sys.stderr, 'make list of missing words ...'
		self.ensureMissingWordList()


#main
optparser = OptionParser(\
    'usage: %prog [mgtl] [OPTION]\n'\
    '       w: make list of missing words\n'\
    '       m: make g2p-model\n'\
    '       t: make transcription of missing words, implies wm\n'\
    '       l: make complete lexicon, implies t\n'\
    '       if none is given, l is assumed'
    )
optparser.add_option("-e", "--encoding", dest="encoding", default="ascii",
		     help="encoding of plain files and resulting bliss lexicon; default is 'ascii'", metavar="ENCODING")
optparser.add_option("-c", "--costa-log-file", dest="costaLogFile", default=None,
		     help="costa log file determining lexicon and missing word list", metavar="FILE")
optparser.add_option("-l", "--lexicon-file", dest="lexiconFile", default=None,
		     help="lexicon to complete", metavar="FILE")
optparser.add_option("-t", "--train-lexicon-file", dest="trainLexiconFile", default=None,
		     help="lexicon to train g2p model", metavar="FILE")
optparser.add_option("-m", "--g2p-model-file", dest="modelFile", default=None,
		     help="g2p model", metavar="FILE")
optparser.add_option("-w", "--missing-word-file", dest="missingWordFile", default=None,
		     help="list of missing words", metavar="FILE")
optparser.add_option("-g", "--g2p-file", dest="missingTransFile", default=None,
		     help="list of missing transcription", metavar="FILE")
optparser.add_option("-o", "--lexicon-out", dest="lexiconOut", default=None,
		     help="write complete lexicon to FILE", metavar="FILE")
optparser.add_option("-n", "--context", dest="context", type='int', default=8,
		     help="context of the g2p-model", metavar="N")
optparser.add_option("-v", "--variants", dest="variants", type='float', default=0.0,
		     help="produces pronunciation variants (see g2p.py --help)", metavar="Q")
optparser.add_option("-f", "--force", dest="force", action='store_true', default=False,
		     help="existing files are overwritten, besides the g2p-model")
optparser.add_option("-F", "--fforce", dest="fforce", action='store_true', default=False,
		     help="all existing files are overwritten")
if len(sys.argv) == 1:
    sys.argv.append('--help')
options, args = optparser.parse_args()
if len(args) == 0:
    args.append('l')

lexMaker = LexiconMaker(options)
lexMaker.run(args[0])
