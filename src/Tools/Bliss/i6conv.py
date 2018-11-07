#!/usr/bin/env python

import sys, re
from optparse import OptionParser, OptionGroup

encodingDescriptionList = [
	['ascii', '646, us-ascii', 'English'],
	['cp037', 'IBM037, IBM039', 'English'],
	['cp424', 'EBCDIC-CP-HE, IBM424', 'Hebrew'],
	['cp437', '437, IBM437', 'English'],
	['cp500', 'EBCDIC-CP-BE, EBCDIC-CP-CH, IBM500', 'Western Europe'],
	['cp737', '', 'Greek'],
	['cp775', 'IBM775', 'Baltic languages'],
	['cp850', '850, IBM850', 'Western Europe'],
	['cp852', '852, IBM852', 'Central and Eastern Europe'],
	['cp855', '855, IBM855', 'Bulgarian, Byelorussian, Macedonian, Russian, Serbian'],
	['cp856', '', 'Hebrew'],
	['cp857', '857, IBM857', 'Turkish'],
	['cp860', '860, IBM860', 'Portuguese'],
	['cp861', '861, CP-IS, IBM861', 'Icelandic'],
	['cp862', '862, IBM862', 'Hebrew'],
	['cp863', '863, IBM863', 'Canadian'],
	['cp864', 'IBM864', 'Arabic'],
	['cp865', '865, IBM865', 'Danish, Norwegian'],
	['cp869', '869, CP-GR, IBM869', 'Greek'],
	['cp874', '', 'Thai'],
	['cp875', '', 'Greek'],
	['cp1006', '', 'Urdu'],
	['cp1026', 'ibm1026', 'Turkish'],
	['cp1140', 'ibm1140', 'Western Europe'],
	['cp1250', 'windows-1250', 'Central and Eastern Europe'],
	['cp1251', 'windows-1251', 'Bulgarian, Byelorussian, Macedonian, Russian, Serbian'],
	['cp1252', 'windows-1252', 'Western Europe'],
	['cp1253', 'windows-1253', 'Greek'],
	['cp1254', 'windows-1254', 'Turkish'],
	['cp1255', 'windows-1255', 'Hebrew'],
	['cp1256', 'windows1256', 'Arabic'],
	['cp1257', 'windows-1257', 'Baltic languages'],
	['cp1258', 'windows-1258', 'Vietnamese'],
	['latin_1', 'iso-8859-1, iso8859-1, 8859, cp819, latin, latin1, L1', 'West Europe'],
	['iso8859_2', 'iso-8859-2, latin2, L2', 'Central and Eastern Europe'],
	['iso8859_3', 'iso-8859-3, latin3, L3', 'Esperanto, Maltese'],
	['iso8859_4', 'iso-8859-4, latin4, L4', 'Baltic languagues'],
	['iso8859_5', 'iso-8859-5, cyrillic', 'Bulgarian, Byelorussian, Macedonian, Russian, Serbian'],
	['iso8859_6', 'iso-8859-6, arabic', 'Arabic'],
	['iso8859_7', 'iso-8859-7, greek, greek8', 'Greek'],
	['iso8859_8', 'iso-8859-8, hebrew', 'Hebrew'],
	['iso8859_9', 'iso-8859-9, latin5, L5', 'Turkish'],
	['iso8859_10', 'iso-8859-10, latin6, L6', 'Nordic languages'],
	['iso8859_13', 'iso-8859-13', 'Baltic languages'],
	['iso8859_14', 'iso-8859-14, latin8, L8', 'Celtic languages'],
	['iso8859_15', 'iso-8859-15', 'Western Europe'],
	['koi8_r', '', 'Russian'],
	['koi8_u', '', 'Ukrainian'],
	['mac_cyrillic', 'maccyrillic', 'Bulgarian, Byelorussian, Macedonian, Russian, Serbian'],
	['mac_greek', 'macgreek', 'Greek'],
	['mac_iceland', 'maciceland', 'Icelandic'],
	['mac_latin2', 'maclatin2, maccentraleurope', 'Central and Eastern Europe'],
	['mac_roman', 'macroman', 'Western Europe'],
	['mac_turkish', 'macturkish', 'Turkish'],
	['utf_16', 'U16, utf16', 'all languages'],
	['utf_16_be', 'UTF-16BE', 'all languages (BMP only)'],
	['utf_16_le', 'UTF-16LE', 'all languages (BMP only)'],
	['utf_7', 'U7', 'all languages'],
	['utf_8', 'U8, UTF, utf8', 'all languages']
	]

# ===========================================================================
def loadTranslationTable(fileName):
    translationTable = {}
    re_line = re.compile(r'^(.*)\s+->\s+(.*?)\s*(#.*)?$', re.UNICODE)
    for line in open(fileName):
	if not line.startswith('#'):
	    m = re_line.match(line)
	    if not m:
		raise ValueError('mapping line must have the form "xxx -> yyy"')
	    if type(eval(m.group(1))) is not int:
		code = ord(eval(m.group(1)))
	    else:
		code = int(eval(m.group(1)))
	    replacement = unicode(eval(m.group(2)))
	    translationTable[code] = replacement
    return translationTable

# ===========================================================================
import codecs

class PythonCodec(codecs.Codec):
    def encode(self, input, errors='strict'):
	return repr(input), len(input)

    def decode(self, input, errors='strict'):
	return eval(input), len(input)

class PythonStreamWriter(PythonCodec, codecs.StreamWriter):
    pass

class PythonStreamReader(PythonCodec, codecs.StreamReader):
    pass

def pythonCodecSearch(name):
    if name == 'py' or name == 'python':
	return repr, eval, PythonStreamReader, PythonStreamWriter
    return None

codecs.register(pythonCodecSearch)

class Printer:
    def __init__(self, stream, errors='strict'):
	self.stream = stream

    def write(self, input):
	self.stream.write(repr(input) + '\n')

# ===========================================================================
import codecs

def main(options, args):
    if options.list:
	print '%-10s %-20s %-40s' % ('ENCODING:', 'ALIAS:', 'REGION:')
	print '-'*60
	for encodingDescription in encodingDescriptionList:
	    print '%-10s %-20s %-40s' % (encodingDescription[0],encodingDescription[1],encodingDescription[2])
	sys.exit(0)

    if not options.from_code and not options.to_code == None:
	print >> sys.stderr, "tty device encoding guessing not implemented yet"
	sys.exit(1)

    inEncoder,  inDecoder,  inStreamReader,  inStreamWriter  = codecs.lookup(options.from_code)
    if options.to_code == 'python':
	outStreamWriter = Printer
    else:
	outEncoder, outDecoder, outStreamReader, outStreamWriter = codecs.lookup(options.to_code)

    if options.translationTableFileName:
	translationTable = loadTranslationTable(options.translationTableFileName)
    else:
	translationTable = {}

    if options.fileOutName:
	fileOut = open(options.fileOutName, 'w')
    else:
	fileOut = sys.stdout
    fileOut = outStreamWriter(fileOut, options.onErrors)

    if args:
	for fname in args:
	    fileIn = open(fname)
	    fileIn = inStreamReader(fileIn)
	    for line in fileIn:
		line = line.translate(translationTable)
		fileOut.write(line)
	    fileIn.close()
    else:
	fileIn = inStreamReader(sys.stdin)
	for line in fileIn:
	    line = line.translate(translationTable)
	    fileOut.write(line)

# ===========================================================================
if __name__ == '__main__':
    optParser = OptionParser(usage="usage: %prog [OPTION...] [FILE...]", version="%prog 0.1")
    optParser.set_description("Convert encoding of given files from one encoding to another.")
    optGroupEnc = OptionGroup( optParser, 'Input/Output format specification' )
    optGroupEnc.add_option(
	"-f", "--from-code", metavar="NAME",
	help="encoding of original text")
    optGroupEnc.add_option(
	"-t", "--to-code", metavar="NAME",
	help="encoding for output")
    optGroupEnc.add_option(
	'', "--map", metavar="FILE",
	dest="translationTableFileName", help="set translation table file name",
	default=None)
    optGroupEnc.add_option(
	'', "--lower", action="store_true", dest="lowerCase",
	help="convert text to lower case",
	default=False)
    optGroupEnc.add_option(
	'', "--upper", action="store_true", dest="upperCase",
	help="convert text to upper case",
	default=False)
    optGroupEnc.add_option(
	'', "--error", metavar="TYPE", dest="onErrors",
	help="set error handling type, default='ignore'",
	default='strict')
    optGroupEnc.add_option(
	"-l", "--list", action="store_true", dest="list",
	help="list all known coded character sets",
	default=False)
    optParser.add_option_group(optGroupEnc)
    optGroupOut = OptionGroup( optParser, 'Output control' )
    optGroupOut.add_option(
	"-o", "--output", dest="fileOutName",
	help="write to FILE instead to standard output", metavar="FILE")
    optParser.add_option_group(optGroupOut)
    optGroup = OptionGroup(
	optParser, 'Note', "With no FILE read from standard input. "+
	"Report bugs to <gollan@cs.rwth-aachen.de>.")
    optParser.add_option_group(optGroup)
    options, args = optParser.parse_args()
    main(options, args)
