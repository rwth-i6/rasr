# -*- coding: iso-8859-1 -*-
import re
import sys
from xml import sax
from xml.sax import saxutils
from miscLib import zopen, zclose

class SimpleXmlParser(sax.handler.ContentHandler):
    """
    Uses the standard sax parser to implement an xml-Parser.
    If no handler is specified at object creation time, 'self'
    is used as handler, otherwise handler must be an instance
    of sax.handler.ContentHandler.
    Two additioanl callback functions are provided:
    startFile(path, encoding)
      'path' is the path of the parsed xml-file and 'encoding' the used encoding
    endFile(path)
      'path' is the path of the parsed xml-file
    """
    def __init__(self, handler = None):
	if handler is None:
	    self.__handler__ = self
	else:
	    self.__handler__ = handler
	self.__default_encoding__ = 'ascii'
	self.__sax_parser__ = sax.make_parser()
	self.__sax_parser__.setFeature(sax.handler.feature_namespaces, 0)
	self.__sax_parser__.setFeature(sax.handler.feature_external_ges, False)
	self.__sax_parser__.setFeature(sax.handler.feature_external_pes, False)
	self.__sax_parser__.setContentHandler(self.__handler__)

    def parse(self, path):
	fd = zopen(path, 'r')
	# determine encoding
	# - assume everthing to be in ascii-encoding until <?xml ...> is found
	# - assume <?xml ...> not to be splitted over several lines
	history = []
	reEncoding = re.compile(r'<\?xml[^>]* encoding="([^"]*)"')
	try:
	    row = fd.next()
	    history.append(row)
	    while row.find('<?xml') == -1:
		row = fd.next()
	except StopIteration:
	    print >> sys.stderr, 'Error: no xml header <?xml ...> found; "' + path + '" is probably not a proper xml-file.'
	    sys.exit(1)
	m = reEncoding.search(row)
	if (m):
	    encoding = m.group(1)
	else:
	    encoding = self.__default_encoding__
	    print >> sys.stderr, 'Warning: no encoding specified, use "' + encoding + '"'
	# parse file
	parser = self.__sax_parser__
	try:
	    self.__handler__.startFile(path, encoding)
	except AttributeError:
	    pass
	for row in history:
	    parser.feed(row)
	del history
	for row in fd:
	    parser.feed(row)
	try:
	    self.__handler__.endFile(path)
	except AttributeError:
	    pass
	zclose(fd)


def parseXml(path, handler):
    """
    Parses the xml-file specified in path.
    'handler' must be an instance of sax.handler.ContentHandler.
    'handler' can support two more callbacks, 'startFile' and 'endFile',
    for details see 'SimpleXmlParser' above.
    """
    parser = SimpleXmlParser(handler)
    parser.parse(path)
