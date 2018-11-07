#!/usr/bin/env python

import sys
import os
from subprocess import Popen, check_call

if sys.platform == "darwin":
	ExecPath = "./nn-trainer.darwin-x86_64-standard"
else:
	ExecPath = "./nn-trainer.linux-x86_64-standard"


def init(**kwargs):
	print("init: %r" % kwargs)
	return Control()


class Control:
	def __init__(self, **kwargs):
		self.seg_counter = 0

	def run_control_loop(self, callback, **kwargs):
		print("run_control_loop: %r, %r" % (callback, kwargs))

	def init_processing(self, **kwargs):
		print("init_processing: %r" % kwargs)

	def process_segment(self, **kwargs):
		self.seg_counter += 1

	def exit(self, **kwargs):
		print("exit: %r" % kwargs)
		print("exit after %i segments" % self.seg_counter)


def getSegmentList(corpusName, segmentList, segmentsInfo, **kwargs):
	print("getSegmentList: %r, ..., %r" % (corpusName, kwargs))
	if not segmentsInfo:
		print("  no segmentsInfo")
	print("  num segments: %i" % len(segmentList))
	for s in segmentList:
		txt = s
		if segmentsInfo: txt += ", info: %r" % segmentsInfo[s]
		print("  %s" % txt)
	return segmentList


if __name__ == "__main__":
	mydir = os.path.dirname(__file__)
	os.chdir(mydir)
	assert os.path.exists(ExecPath), "%s/%s not found" % (mydir, ExecPath)
	assert len(sys.argv) >= 2, "please provide corpus file as first argument"
	corpusFile = sys.argv[1]  # e.g. /u/tuske/work/ASR/switchboard/corpus/xml/train.corpus.gz
	assert os.path.exists(corpusFile), "corpus file %r not found" % corpusFile
	args = [
		ExecPath,
		"--*.action=python-control",
		"--*.pymod-name=python_segment_ordering_demo",
		"--*.python-control-loop-type=iterate-corpus",
		"--*.extract-features=false",
		"--*.corpus.file=%s" % corpusFile,
		"--*.python-segment-order=true",
		"--*.python-segment-order-pymod-path=.",
		"--*.python-segment-order-pymod-name=python_segment_ordering_demo",
		"--*.python-segment-order-with-segment-info=true",
		] + sys.argv[2:]
	print("Call: %r" % args)
	check_call(args)

