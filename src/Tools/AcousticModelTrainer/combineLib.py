# -*- coding: iso-8859-1 -*-

import os, os.path, tempfile


def mkTmp():
    tmpFd, tmpFile = tempfile.mkstemp()
    os.close(tmpFd)
    return tmpFile

def combine(srcFiles, trgFile, combineFcn):
    assert len(srcFiles) > 1
    if len(srcFiles) > 3:
	nextStep = []
	while len(srcFiles) > 1:
	    target = mkTmp()
	    combineFcn(srcFiles.pop(), srcFiles.pop(), target)
	    nextStep.append(target)
	if srcFiles:
	    target = mkTmp()
	    source1 = nextStep.pop()
	    combineFcn(source1, srcFiles.pop(), target)
	    os.unlink(source1)
	    nextStep.append(target)
	while len(nextStep) > 2:
	    thisStep = nextStep
	    nextStep = []
	    while len(thisStep) > 1:
		target = mkTmp()
		source1 = thisStep.pop()
		source2 = thisStep.pop()
		combineFcn(source1, source2, target)
		os.unlink(source1)
		os.unlink(source2)
		nextStep.append(target)
	    if thisStep:
		nextStep.append(thisStep.pop())
	assert len(nextStep) == 2
	combineFcn(nextStep[0], nextStep[1], trgFile)
	os.unlink(nextStep[0])
	os.unlink(nextStep[1])
    else:
	if len(srcFiles) == 3:
	    tmp = mkTmp()
	    combineFcn(srcFiles.pop(), srcFiles.pop(), tmp)
	    combineFcn(srcFiles.pop(), tmp, trgFile)
	    os.unlink(tmp)
	else:
	    combineFcn(srcFiles.pop(), srcFiles.pop(), trgFile)
