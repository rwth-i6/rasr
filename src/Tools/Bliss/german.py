class UmlautRepair:
    """
    German uses has four extra characters which are only avaiable on
    German keyboards.  As a result people often replace them by latin
    equivalents.

    The character seqeuences "ae", "oe", "ue" and "ss" are replaced by
    "ä", "ö", "ü" and "ß" respectively, if the so produced word occurs
    more frequently than the original form.
    """

    umlautReplacements = [
	(re.compile('ae'), u'ä'),
	(re.compile('oe'), u'ö'),
	(re.compile('ue'), u'ü'),
	(re.compile('Ae'), u'Ä'),
	(re.compile('Oe'), u'Ö'),
	(re.compile('Ue'), u'Ü'),
	(re.compile('ss'), u'ß')]

    def __init__(self, counts):
	self.rawCounts = counts

    def __call__(self, tokens):
	result = []
	for token in tokens:
	    candidate = token
	    for r, s in self.umlautReplacements:
		candidate = r.sub(s, candidate)
	    if candidate != token:
		if self.rawCounts[candidate] > self.rawCounts[token]:
		    token = candidate
	    result.append(token)
	return result
