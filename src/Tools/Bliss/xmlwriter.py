__author__    = 'Maximilian Bisani'
__version__   = '$Revision$'
__date__      = '$Date$'
__copyright__ = 'Copyright (c) 2002-2005'

import codecs

class XmlWriter:
    def __init__(self, file):
	self.path = []
	self.file = file
	self.indentation = 2
	self.margin = 78
	self.setEncoding('UTF-8')

    def setEncoding(self, encode):
	self.encoding = encode
	self.writer = codecs.getwriter(self.encoding)(self.file)

    def write(self, data):
	self.writer.write(data)

    def begin(self):
	self.write(u'<?xml version="1.0" encoding="%s"?>\n' %
		   self.encoding)

    def end(self):
	assert len(self.path) == 0
	pass

    def setMargin(self, margin):
	self.margin = margin

    def setIndentation(self, amount):
	self.indentation = amount

    def indentStr(self):
	return ' ' * (len(self.path) * self.indentation)

    def escape(self, w):
	return w.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def formTag(self, element, attr=[]):
	return self.escape(' '.join([element] + [ '%s="%s"' % kv for kv in attr]))

    def open(self, element, atts={}, **args):
	atts = atts.items() + args.items()
	atts = filter(lambda (k, v): v is not None, atts)
	self.write(self.indentStr() + '<' + self.formTag(element, atts) + '>\n')
	self.path.append(element)

    def empty(self, element, atts = {}, **args):
	atts = atts.items() + args.items()
	atts = filter(lambda (k, v): v is not None, atts)
	self.write(self.indentStr() + '<' + self.formTag(element, atts) + '/>\n')

    def close(self, element):
	assert element == self.path[-1]
	del self.path[-1]
	self.write(self.indentStr() + '</' + element + '>\n')

    def openComment(self):
	self.write('<!--\n')
	self.path.append('<!--')

    def closeComment(self):
	assert self.path[-1] == '<!--'
	del self.path[-1]
	self.write('-->\n')

    formatRaw = 0
    formatIndent = 1
    formatBreakLines = 2
    formatFill = 3

    def fillParagraph(self, w):
	indentStr = self.indentStr()
	ll = []
	l = [] ; n = len(indentStr)
	for a in w.split():
	    if n + len(a) < self.margin:
		n = n + len(a) + 1
		l.append(a)
	    else:
		ll.append(indentStr + ' '.join(l))
		l = [a] ; n = len(indentStr) + len(a)
	if len(l) > 0:
	    ll.append(indentStr + ' '.join(l))
	return ll

    def cdata(self, w, format = formatFill):
	if 'u<!--' in self.path:
	    w = w.replace('--', '=') # comment must not contain double-hyphens
	if format == self.formatRaw:
	    out = [ w ]
	elif format == self.formatIndent:
	    indentStr = self.indentStr()
	    out = [ indentStr + line for line in w.split('\n') ]
	elif format == self.formatBreakLines:
	    out = [ self.fillParagraph(line) for line in w.split('\n') ]
	    out = reduce(operator.add, out)
	elif format == self.formatFill:
	    out = self.fillParagraph(w)
	self.write('\n'.join(out) + '\n')

    def formatted_cdata(self, s):
	for w in s.split('\\n'):
	    self.cdata(w, self.formatFill)

    def comment(self, comment):
	comment = comment.replace('--', '=') # comment must not contain double-hyphens
	self.cdata('<!-- ' + comment + ' -->')

    def element(self, element, cdata=None, atts={}, **args):
	if cdata is None:
	    self.empty(element, atts, **args)
	else:
	    atts = atts.items() + args.items()
	    atts = filter(lambda (k, v): v is not None, atts)
	    s = self.indentStr() \
		+ '<' + self.formTag(element, atts) + '>' \
		+ cdata \
		+ '</' + element + '>'
	    if len(s) <= self.margin:
		self.write(s + '\n')
	    else:
		apply(self.open, (element,), args)
		self.cdata(cdata)
		self.close(element)
