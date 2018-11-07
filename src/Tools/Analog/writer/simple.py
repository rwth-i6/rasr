"""
Analog Plug-in for reating simple formats such as recognized sentences.
"""

__version__   = '$Revision$'
__date__      = '$Date$'


import string
from analog import Writer

class WriteList(Writer):
    def __init__(self, options, fieldName):
	super(WriteList, self).__init__(options)
	self.fieldName = fieldName

    def __call__(self, file, data):
	for record in data:
	    print >> file, record[self.fieldName]

class RecognizedSentences(WriteList):
    id = 'recognized'
    defaultPostfix = '.rec'

    def __init__(self, options):
	super(RecognizedSentences, self).__init__(options, 'recognized')

class ReferenceSentences(WriteList):
    id = 'reference'
    defaultPostfix = '.ref'

    def __init__(self, options):
	super(ReferenceSentences, self).__init__(options, 'reference')
