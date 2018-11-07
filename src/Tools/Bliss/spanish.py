# -*- coding: ISO-8859-1 -*-

"""
Text normalisation and conditioning function for Spanish.
"""

__version__   = '$Revision$'
__date__      = '$Date$'


import re, sys

monthNames = [
    u'enero',
    u'febrero',
    u'marzo',
    u'abril',
    u'mayo',
    u'junio',
    u'julio',
    u'agosto',
    u'septiembre',
    u'octubre',
    u'noviembre',
    u'diciembre' ]


class NumberConverter:
    _maxNumberStr = str(sys.maxint)
    _maxNumber    = sys.maxint

    def safeInt(self, str):
	length = len(str)
	if (length > 10) or ((length == 10) and (str > self._maxNumberStr)):
	    return self._maxNumber
	else:
	    return int(str)

    # months
    monthList = [ None ] + monthNames

    # month: abbr. to full name
    monthDict = {
	'ene' : u'enero,',
	'feb' : u'febrero',
	'mar' : u'marzo',
	'abr' : u'abril',
	'may' : u'mayo',
	'jun' : u'junio',
	'jul' : u'julio',
	'ago' : u'agosto',
	'sep' : u'septiembre',
	'cot' : u'octubre',
	'nov' : u'noviembre',
	'dic' : u'diciembre'
	}

    # currencies: singular and plural
    currencyDictS = {
	'$'    : u'd\xf3lar',
	'USD'  : u'd\xf3lar',
	'DEM'  : u'marco',
	'GBP'  : u'libra',
	u'\xa3': u'libra',
	'FRF'  : u'franco',
	'JPY'  : u'yen',
	'ecu'  : u'ecu',
	'EUR'  : u'euro',
	'cent' : u'c\xe9ntimo'
	}

    currencyDictP = {
	'$'    : u'd\xf3lares',
	'USD'  : u'd\xf3lares',
	'DEM'  : u'marcos',
	'GBP'  : u'libras',
	u'\xa3': u'libras',
	'FRF'  : u'francos',
	'JPY'  : u'yens',
	'ecu'  : u'ecus',
	'EUR'  : u'euros',
	'cent' : u'c\xe9ntimos'
	}

    # numbers
    numberList = [
	u'cero',
	u'uno',
	u'dos',
	u'tres',
	u'cuatro',
	u'cinco',
	u'seis',
	u'siete',
	u'ocho',
	u'nueve',
	u'diez',
	u'once',
	u'doce',
	u'trece',
	u'catorce',
	u'quince',
	u'dieciséis',
	u'diecisiete',
	u'dieciocho',
	u'diecinueve',
	u'veinte',
	u'veintiuno',
	u'veintidós',
	u'veintitrés',
	u'veinticuatro',
	u'veinticinco',
	u'veintiséis',
	u'veintisiete',
	u'veintiocho',
	u'veintinueve'
	]

    # numbers: in roman digits
    romanDict = {
#	'I'      : '1',
	'II'     : '2',
	'III'    : '3',
	'IV'     : '4',
	'V'      : '5',
	'VI'     : '6',
	'VII'    : '7',
	'VIII'   : '8',
	'IX'     : '9',
	'X'      : '10',
	'XI'     : '11',
	'XII'    : '12',
	'XIII'   : '13',
	'XIV'    : '14',
	'XV'     : '15',
	'XVI'    : '16',
	'XVII'   : '17',
	'XVIII'  : '18',
	'XIX'    : '19',
	'XX'     : '20',
	'XXI'    : '21',
	'XXII'   : '22',
	'XXIII'  : '23',
	'XXIV'   : '24',
	'XXV'    : '25',
	'XXVI'   : '26',
	'XXVII'  : '27',
	'XXVIII' : '28',
	'XXIX'   : '29',
	'XXX'    : '30'
	}

    # numbers: 30, 40, ... 90
    decList = [
	u'',
	u'',
	u'',
	u'treinta',
	u'cuarenta',
	u'cincuenta',
	u'sesenta',
	u'setenta',
	u'ochenta',
	u'noventa'
	]

    # numbers: 10^0, 10^1, ... 10^6, singular and plural
    powerListS = [
	u'uno',
	u'diez',
	u'ciento',
	u'mil',
	u'diez mil',
	u'ciento mil',
	u'mill\xf3n'
	]

    powerListP = [
	u'uno',
	u'diez',
	u'cientos',
	u'mil',
	u'diez mil',
	u'ciento mil',
	u'millones'
	]



    # o/O in numbers -----------------------------------------------------------
    # example: 15oo gmt, 15:oo,
    _oToZeroRE = re.compile( \
	 r'(?:(?<=\d)[oO]+(?=\D?\d|[.,\s]))|' \
	 r'(?:(?<=\d\D)[oO]+(?=[\d.,\s]))|' \
	 r'(?:(?<=\d)[oO]{2,})|' \
	 r'(?:(?<=\d\D)[oO]{2,})' \
	 , re.UNICODE)

    def _oToZeroSub(self, m):
	return '0'*len(m.group(0))

    def convertO(self, s):
	s = ' ' + s.strip() + ' '
	t = self._oToZeroRE.sub(self._oToZeroSub, s)
	while t != s:
	    s = t
	    t = self._oToZeroRE.sub(self._oToZeroSub, s)
	return t


    # formatted numbers --------------------------------------------------------
    # example: 1.000, 2,345,456
    _long1RE = re.compile(r'(?<=\D)' \
			 r'\d{1,3}(?:\.\d{3})+' \
			 r'(?=\D)' \
			  , re.UNICODE)

    def _long1Sub(self, m):
	return m.group(0).replace('.', '')

    _long2RE = re.compile(r'(?<=\D)' \
			 r'(?:' \
			 r'\d{1,3}(?:,\d{3})+(?=\.\d)' \
			 r'|\d{1,3}(?:,\d{3}){2,}(?=\D)'   \
			 r')' \
			  , re.UNICODE)

    def _long2Sub(self, m):
	return m.group(0).replace(',', '')

    def eliminateSeperators(self, s):
	s = ' ' + s.strip() + ' '
	s = self._long1RE.sub(self._long1Sub, s)
	s = self._long2RE.sub(self._long2Sub, s)
	return s


    # formatted time strings ---------------------------------------------------
    # example: 12H05, 2:34:56, 0123GMT
    _gmtTimeRE  = re.compile( \
	r'(?<=\D)' \
	r'(?P<h>\d{1,2})(?P<sep>[.,:Hh]?)(?P<m>\d{2})?(?:(?P=sep)(?P<s>\d{2}))?' \
	r'\s*(?P<prefix>GMT|gmt)' \
	, re.UNICODE)
    _time1RE    = re.compile( \
	r'(?<=.[^\d:]|\D:)' \
	r'(?P<h>\d{1,2})(?P<sep>[.,:Hh])(?P<m>\d{2})(?:(?:(?P=sep)|:)(?P<s>\d{2}))?' \
	r'\s*(?P<prefix>horas)' \
	, re.UNICODE)
    _time2RE    = re.compile( \
	r'(?<=.[^\d:]|\D:)' \
	r'(?P<h>\d{1,2})(?P<sep>[:Hh])(?P<m>\d{2})(?:(?:(?P=sep)|:)(?P<s>\d{2}))?' \
	r'(?=[^\d:]|:\D)' \
	, re.UNICODE)

    def _makeTime(self, m):
	time = ''
	hour = int(m.group('h'))
	if hour == 1:
	    time += 'un hora'
	else:
	    time += m.group('h') + ' horas'
	if m.group('m') is not None:
	    min = int(m.group('m'))
	    if min == 1:
		time += ' y un minuto'
	    else:
		time += ' y ' + m.group('m') + ' minutos'
	if m.group('s') is not None:
	    sec = int(m.group('s'))
	    if sec == 1:
		time += ' y un segundo'
	    else:
		time += ' y ' + m.group('s') + ' segundos'
	return time

    def _makeGmtTime(self, m):
	time = self._makeTime(m)
	time += ' GMT '
	return time

    def convertTimeData(self, s):
	s = ' ' + s.strip() + ' '
	s = self._gmtTimeRE.sub(self._makeGmtTime, s)
	s = self._time1RE.sub  (self._makeTime, s)
	s = self._time2RE.sub  (self._makeTime, s)
	return s


    # formatted date strings ---------------------------------------------------
    # example: 1: 26/11/76, 24/12/
    #          2: 16Nov95
    #          3: 30/11 - bla bla, 12/11: bla bla
    _date1RE = re.compile( \
	r'(?<=\s)' \
	r'(?P<day>\d{1,2})(?P<sep>[\-/.])(?P<month>\d{1,2})(?P=sep)(?P<year>\d{2}(?:\d{2})?)?' \
	r'(?=\s|\D\D)' \
	, re.UNICODE)
    _date2RE = re.compile( \
	r'(?<=\s)' \
	r'(?P<day>\d{1,2}) *(?P<month>[^\W\d_]{3}) *(?P<year>\d{2}(?:\d{2})?)?' \
	r'(?=\s|\D\D)' \
	, re.UNICODE)
    _date3RE = re.compile( \
	r'(?<=\s)' \
	r'(?P<day>\d{1,2})(?P<sep>[\-/.])(?P<month>\d{1,2})' \
	r'(?=:\D|\s*-\D)' \
	, re.UNICODE)
    _date4RE = re.compile( \
	r'(?<=\sel\s)' \
	r'(?P<day>\d{1,2})(?P<sep>[\-/.])(?P<month>\d{1,2})' \
	r'(?=\s|\D\D)' \
	, re.UNICODE)

    def _makeDate1(self, m):
	month = int(m.group('month'))
	if month < 1 or month > 12:
	    return m.group(0)
	else:
	    date = m.group('day') + ' ' + self.monthList[month]
	    if m.group('year') != None:
		year = m.group('year')
		if len(year) == 2:
		    date += ' 19' + year
		else:
		    date += ' ' + year
	    return date

    def _makeDate2(self, m):
	month = m.group('month').lower()
	if month not in self.monthDict:
	    return m.group(0)
	else:
	    date = m.group('day') + ' ' + self.monthDict[month]
	    if m.group('year') != None:
		year = m.group('year')
		if len(year) == 2:
		    date += ' 19' + year
		else:
		    date += ' ' + year
	    return date

    def _makeDate3(self, m):
	month = int(m.group('month'))
	if month < 1 or month > 12:
	    return m.group(0)
	else:
	    return m.group('day') + ' ' + self.monthList[month]

    def convertDateSpecification(self, s):
	s = ' ' + s.strip() + ' '
	s = self._date1RE.sub(self._makeDate1, s)
	s = self._date2RE.sub(self._makeDate2, s)
	s = self._date3RE.sub(self._makeDate3, s)
	s = self._date4RE.sub(self._makeDate3, s)
	return s


    # formatted currencies -----------------------------------------------------
    # $100, 12,34 USD
    _currency1RE = re.compile( \
	r'\W' \
	r'(?P<currency>\$|USD|DEM|GBP|\xa3|FRF|JPY|ecu|EUR)\s?(?P<high>\d+)(?:[.,](?P<low>\d+))?' \
	, re.UNICODE)
    _currency2RE = re.compile( \
	r'(?P<high>\d+)(?:[.,](?P<low>\d+))?\s?(?P<currency>\$|USD|DEM|GBP|\xa3|FRF|JPY|ecu|EUR)' \
	r'(?=\W|\d)' \
	, re.UNICODE)

    def _makeCurrency(self, m):
	high = self.safeInt(m.group('high'))
	low  = 0
	if (m.group('low') is not None):
	    low = self.safeInt(m.group('low'))
	currency = ''

	if (high > 0) or (low == 0):
	    str = m.group('high') + ' '
	    if high == 1:
		currency += self.currencyDictS[m.group('currency')]
	    else:
		currency += self.currencyDictP[m.group('currency')]
	    if low > 0:
		if len(m.group('low')) == 1:
		    currency += ' ' + m.group('low') + '0'
		else:
		    currency += ' ' + m.group('low')
	else:
	    currency = m.group('low') + ' '
	    if low == 1:
		currency += self.currencyDictS['cent']
	    else:
		currency += self.currencyDictP['cent']
	return currency

    def convertCurrency(self, s):
	s = ' ' + s.strip() + ' '
	s = self._currency1RE.sub(self._makeCurrency, s)
	s = self._currency2RE.sub(self._makeCurrency, s)
	return s


    # ordinals -------------------------------------------------------------
    # example:  el 25º aniversario

    _ordinalRE = re.compile(
	r'(?<=\s)' \
	r'(\d+)([ºª])'\
	r'(?=\D)' \
	, re.UNICODE)

    ordinalList = [
	None,
	u'primer?',
	u'segund?',
	u'tercer?',
	u'cuart?',
	u'quint?',
	u'sext?',
	u'séptim?',
	u'octav?',
	u'noven?',
	u'décim?',
	u'undécim?',       # onceav?, décimo primer?
	u'duodécim?',      # doceav?, décimo segund?
	u'décimo tercer?', # treceav?
	u'décimo cuart?',  # catorceav?
	u'décimo quint?',  # quinceav?
	u'décimo sext?',   # dieciseisav?
	u'décimo séptim?', # diecisieteav?
	u'décimo octav?',  # dieciochoav?
	u'décimo noven?',  # deicinueveav?
	u'vigésim?'        # veinteav?
	]

    def _makeOrdinal(self, m):
	i = self.safeInt(m.group(1))
	if i <= 0:
	    return ' '
	elif i < len(self.ordinalList):
	    text = self.ordinalList[i] + ' '
	    if m.group(2) == u'º':
		return text.replace('?', 'o')
	    elif m.group(2) == u'ª':
		return text.replace('?', 'a')
	else:
	    return self._makeNumber(i) + ' '

    def convertOrdinal(self, s):
	"""
	Convert ordinal numbers.
	Example:  "el 25º aniversario"

	On Tuesday 08 March 2005 14:38, Davi Vilar wrote:
	> On Tue, Mar 08, 2005 at 02:03:24PM +0100, Maximilian Bisani wrote:
	> > Hallo David!
	> >
	> > Vermute ich richtig, dass "nº" für "numero" steht?
	>
	> Ja, das ist richtig.
	>
	> > Und was fange ich mit [0-9]+º an?  (z.B. "el 25º aniversario")
	> > Sind das immer Ordinalzahlen?
	>
	> Ja, die sind Ordinalzahlen, die Aussprache ist aber nicht so einfach
	> zu beschreiben und eigentlich machen viele (spanischen) Leute das
	> falsch. Falls es dir hilft, eine Liste von den ersten 20. Die erste
	> Spalte ist der Zahl, die zweite die richtige Aussprache, die Dritte
	> wie oft (falsch) ausgesprochen wird.
	>
	>  1º primero
	>  2º segundo
	>  3º tercero
	>  4º cuarto
	>  5º quinto
	>  6º sexto
	>  7º séptimo
	>  8º octavo
	>  9º noveno
	> 10º décimo
	> 11º undécimo  onceavo, décimo primero
	> 12º duodécimo  doceavo, décimo segundo
	> 13º décimo tercero  treceavo
	> 14º décimo cuarto  catorceavo
	> 15º décimo quinto  quinceavo
	> 16º décimo sexto  dieciseisavo
	> 17º décimo séptimo  diecisieteavo
	> 18º décimo octavo  dieciochoavo
	> 19º décimo noveno  deicinueveavo
	> 20º vigésimo  veinteavo
	>
	> Danach sagt man normalerweiser einfach der Zahl (in deinem Beispiel:
	> "el veinticinco aniversario"), aber es ist auch möglich daß sie
	> weiter falsch machen (mir der -avo Endung).

	On Tuesday 08 March 2005 18:20, David Vilar wrote:
	> On Tue, Mar 08, 2005 at 03:22:26PM +0100, Maximilian Bisani wrote:
	> > Dann gibt es auch nch das lustige Zeichen "ª"
	> > in "la 7ª Asamblea Parlamentaria Conjunta".  Ich rate, dass das die
	> > Fimunium-Varainte ist.  Kann ich in Deiner Liste einfach "o" dorch "a"
	> > ersetzen?
	>
	> Ja, das geht. Allerdings, in denen mit zwei Wörter ("décimo
	> tercero") wird nur das zweite Wort gewechselt ("décimo segunda").
	"""

	s = ' ' + s.strip() + ' '
	s = self._ordinalRE.sub(self._makeOrdinal, s)
	return s


    # floats -------------------------------------------------------------------
    _floatRE = re.compile( \
	r'(?<=\D[.,\-:]|.[^.,\-:\d])' \
	r'(\d+)([.,])(\d+)' \
	r'(?=[.,\-:]\D|[^.,\-:\d].)' \
	, re.UNICODE)

    def _makeFloat(self, m):
	float = m.group(1)
	low = self.safeInt(m.group(3))
	if low > 0:
	    if m.group(2) == '.':
		float += ' punto '
	    else:
		float += ' coma '
	    if len(m.group(3)) < 4:
		float += m.group(3)
	    else:
		for i in m.group(3):
		    float += i + ' '
		float = float[:-1]
	return float

    def convertFloat(self, s):
	s = ' ' + s.strip() + ' '
	s = self._floatRE.sub(self._makeFloat, s)
	return s


    # numbers (serials, ids, versions, etc.) -----------------------------------
    # example: 2.2.1, 2.95.4-3
    _otherRE = re.compile(r'(?:\d+[.,\-/:])+\d+', re.UNICODE)

    def _makeOther(self, m):
	s = m.group(0)
	s = s.replace('.', ' ')
	s = s.replace(',', ' ')
	s = s.replace('-', ' ')
	s = s.replace('/', ' ')
	s = s.replace(':', ' ')
	return s

    def convertOther(self, s):
	s = ' ' + s.strip() + ' '
	s = self._otherRE.sub(self._makeOther, s)
	return s


    # roman to digits ----------------------------------------------------------
    # example: IX. V
    def convertRoman(self, s):
	s = ' ' + s.strip() + ' '
	for roman, arabic in self.romanDict.iteritems():
	    s = s.replace(' ' + roman + ' ', ' ' + arabic + ' ')
	    s = s.replace(' ' + roman + '.', ' ' + arabic + ' ')
	return s


    # digits to strings --------------------------------------------------------
    # example: 1234567890
    _intRE = re.compile( \
	r'(?<=\D)' \
	r'(?P<number>\d+)' \
	r'(?=\D)' \
	, re.UNICODE)

    def _makeNumber(self, i):
	number = ''
	if i < 30:
	    number += self.numberList[i]
	elif i < 100:
	    i_div = i / 10
	    i_mod = i % 10
	    number = self.decList[i_div]
	    if i_mod > 0:
		number += ' y ' + self.numberList[i_mod]
	elif i < 1000:
	    i_div = i / 100
	    i_mod = i % 100
	    if i_div == 1:
		number = self.powerListS[2]
	    else:
		number = self.numberList[i_div] + ' ' + self.powerListP[2]
	    if i_mod > 0:
		number += ' ' + self._makeNumber(i_mod)
	elif i < 1000000:
	    i_div = i / 1000
	    i_mod = i % 1000
	    if i_div == 1:
		number = self.powerListS[3]
	    else:
		number = self._makeNumber(i_div) + ' ' + self.powerListP[3]
	    if i_mod > 0:
		number += ' ' + self._makeNumber(i_mod)
	else:
	    i_div = i / 1000000
	    i_mod = i % 1000000
	    if i_div == 1:
		number = self.powerListS[6]
	    else:
		number = self._makeNumber(i_div) + ' ' + self.powerListP[6]
	    if i_mod > 0:
		number += ' ' + self._makeNumber(i_mod)
	return number

    def _makeInt(self, m):
	number = m.group('number')
	length = len(number)
	suffix = ' '

	while (length > 10) or ((length == 10) and (number > self._maxNumberStr)):
	    suffix += self.powerListP[6] + ' ' + self._makeNumber(int(number[-6:])) + ' '
	    number = number[:-6]
	    length -= 6

	str = self._makeNumber(int(number))
	if (len(number) > 6) and (len(number) % 6 == 1) and (number[0] == '1'):
	    str = 'un ' + str
	return ' ' + str + suffix

    def convertNumber(self, s):
	s = ' ' + s.strip() + ' '
	s = self._intRE.sub(self._makeInt, s)
	return s


    # --------------------------------------------------------------------------
    def __call__(self, s):
	s = ' '.join(s)
	s = self.convertO(s)
	s = self.eliminateSeperators(s)
	s = self.convertTimeData(s)
	s = self.convertDateSpecification(s)
	s = self.convertCurrency(s)
	s = self.convertOrdinal(s)
	s = self.convertFloat(s)
	s = self.convertOther(s)
	s = self.convertRoman(s)
	s = self.convertNumber(s)
	return s.split()


# ===========================================================================

letters = [
    (line.split()[0], tuple(line.split()[1:]))
    for line in u"""\
	A       " a
	B       " b e
	C       " T e
	D       " d e
	E       " e
	F       " e . f e
	G       " x e
	H       " a . tS e
	I       " i
	J       " x o . t a
	K       " k a
	L       " e . l e
	M       " e . m e
	N       " e . n e
	Ñ       " e . J e
	O       " o
	P       " p e
	Q       " k u
	R       " e . rr e
	S       " e . s e
	T       " t e
	U       " u
	V       " u . B e
	W       " u . B e . " D o . B l e
	X       " e . k i s
	Y       " i . " g r j e . G a
	Z       " T e . t a                      """.split('\n') ]
