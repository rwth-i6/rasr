#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os, os.path, sys, tempfile
from combineLib import combine, mkTmp


def combineScatterMatricesAccus(srcFile1, srcFile2, trgFile):
    cmd = 'acoustic-model-trainer ' \
	  '--*.log.channel=stdout '\
	  '--*.warning.channel=stderr '\
	  '--*.error.channel=stderr '\
	  '--*.action=combine-scatter-matrix-accumulators '\
	  '--*.scatter-matrix-estimator.old-accumulator-file=%s '\
	  '--*.scatter-matrix-estimator-to-combine.old-accumulator-file=%s '\
	  '--*.new-accumulator-file=%s ' \
	  % (srcFile1, srcFile2, trgFile)
    print cmd
    os.system(cmd)
    assert os.path.exists(trgFile)
    print

def estimateScatterMatrices(accuFile, withinFile, betweenFile):
    cmd = 'acoustic-model-trainer ' \
	  '--*.log.channel=stdout '\
	  '--*.warning.channel=stderr '\
	  '--*.error.channel=stderr '\
	  '--*.action=estimate-scatter-matrices-from-accumulator '\
	  '--*.old-accumulator-file=%s' % accuFile
    if withinFile:
	cmd += ' --*.within-class-scatter-matrix-file=%s' % withinFile
    if betweenFile:
	cmd += ' --*.between-class-scatter-matrix-file=%s' % betweenFile
    print cmd
    os.system(cmd)
    if withinFile:
	assert os.path.exists(withinFile)
    if betweenFile:
	assert os.path.exists(betweenFile)
    print


if __name__ == '__main__':
    from optparse import OptionParser
    from ioLib import iterLines
    optparser = OptionParser( \
	usage= \
	'usage:\n %prog [OPTION] <scatter-matrix-accu-file>*\n')
    optparser.add_option("-l", "--mix-list", dest="fileList", default=None,
			 help="file listing source scatter-matrix-accu-files", metavar="FILE")
    optparser.add_option("-b", "--between-class-scatter-matrix", dest="betweenOut", default=None,
			 help="between-class-scatter-matrix", metavar="FILE")
    optparser.add_option("-w", "--within-class-scatter-matrix", dest="withinOut", default=None,
			 help="within-class-scatter-matrix", metavar="FILE")
    opts, args = optparser.parse_args()
    srcFiles=[]
    if opts.fileList:
	srcFiles.extend( [ f for f in iterLines(opts.fileList) ] )
    srcFiles.extend(args)
    tmpFile = mkTmp()
    combine(srcFiles, tmpFile, combineScatterMatricesAccus)
    estimateScatterMatrices(tmpFile, opts.withinOut, opts.betweenOut)
