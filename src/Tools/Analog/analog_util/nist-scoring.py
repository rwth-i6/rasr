#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os
import sys
from optparse import OptionParser
import tempfile

optparser = OptionParser( \
    usage= \
    'usage: %prog [OPTION] log-files [-- sclite options]\n' \
    '       output nist-score of hypotheses against the references in the log-file\n'
    )

optparser.add_option("-g", "--glm-file", dest="glm", default=None,
                    help="hypotheses and references are filtered according to glm-file", metavar="FILE")
optparser.add_option("-s", "--spoken-file", dest="ref", default=None,
                     help="reference file in stm format", metavar="FILE")
optparser.add_option("-r", "--recognized-file", dest="rec", default=None,
                     help="hypotheses file in ctm format", metavar="FILE")

if len(sys.argv) == 1:
    sys.argv.append('--help')

scliteOptions = ''
try:
    scliteOptions = ' '.join(sys.argv[sys.argv.index('--')+1:])
    del sys.argv[sys.argv.index('--'):]
except:
    scliteOptions = '-O "-" -o sum'
options, args = optparser.parse_args()

ref = open(options.ref, 'r')
rec = open(options.rec, 'r')

refFiltered = tempfile.NamedTemporaryFile()
recFiltered = tempfile.NamedTemporaryFile()
if options.glm == None:
    os.system('cat ' + ref.name + '| sort +0 -1 +1 -2 +3nb -4 > ' + refFiltered.name)
    os.system('cat ' + rec.name + '| sort +0 -1 +1 -2 +2nb -3 > ' + recFiltered.name)
else:
    os.system('cat ' + ref.name + '| sort +0 -1 +1 -2 +3nb -4 | csrfilt.sh -i stm ' + options.glm + ' > ' + refFiltered.name)
    os.system('cat ' + rec.name + '| sort +0 -1 +1 -2 +2nb -3 | csrfilt.sh -i ctm ' + options.glm + ' > ' + recFiltered.name)

# score
# -F: false starts
os.system('sclite -F -r ' + refFiltered.name + ' stm -h ' + recFiltered.name + ' ctm ' + scliteOptions)
refFiltered.close()
recFiltered.close()
