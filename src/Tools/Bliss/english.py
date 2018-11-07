#-*- coding: ISO-8859-1 -*-

import re, sys

monthNames = [
    u'January',
    u'February',
    u'March',
    u'April',
    u'May',
    u'June',
    u'July',
    u'August',
    u'September',
    u'October',
    u'November',
    u'December' ]


class NumberConverter:

    # months
    monthList = [ None ] + monthNames

    # month: abbr. to full name
    monthDict = {
	'jan' : u'January',
	'feb' : u'February',
	'mar' : u'March',
	'apr' : u'April',
	'may' : u'May',
	'jun' : u'June',
	'jul' : u'July',
	'aug' : u'August',
	'sep' : u'September',
	'oct' : u'October',
	'nov' : u'November',
	'dec' : u'December'
	}

    # currencies: singular and plural
    currencyDictS = {
	'$'        : u'dollar',
	'USD'      : u'dollar',
	'US$'      : u'dollar',
	'DEM'      : u'deutsch_mark',
	'GBP'      : u'pound',
	u'\xa3'    : u'pound',
	'FF'       : u'franc',
	'FRF'      : u'franc',
	'JPY'      : u'yen',
	'ECU'      : u'ecu',
	'ecu'      : u'ecu',
	'EUR'      : u'euro',
	'cent'     : u'cent'
	}

    currencyDictP = {
	'$'        : u'dollars',
	'USD'      : u'dollars',
	'US$'      : u'dollars',
	'DEM'      : u'deutsch_marks',
	'GBP'      : u'pounds',
	u'\xa3'    : u'pounds',
	'FF'       : u'francs',
	'FRF'      : u'francs',
	'JPY'      : u'yens',
	'ECU'      : u'ecu',
	'ecu'      : u'ecu',
	'EUR'      : u'euros',
	'cent'     : u'cents'
	}

    currencyAbbr = {
	'm'      : u'million',
	'bn'     : u'billion',
	'million': u'million',
	'billion': u'billion'
	}

    # numbers
    numberList = [
	u'O',
	u'one',
	u'two',
	u'three',
	u'four',
	u'five',
	u'six',
	u'seven',
	u'eight',
	u'nine',
	u'ten',
	u'eleven',
	u'twelve',
	u'thirteen',
	u'fourteen',
	u'fifteen',
	u'sixteen',
	u'seventeen',
	u'eighteen',
	u'nineteen'
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
	u'twenty',
	u'thirty',
	u'forty',
	u'fifty',
	u'sixty',
	u'seventy',
	u'eighty',
	u'ninety'
	]

    # numbers: 10^0, 10^1, ... 10^6, singular and plural
    powerList = [
	u'one',
	u'ten',
	u'hundred',
	u'thousand',
	u'ten thousand',
	u'hundred thousand',
	u'million'
	]

    # enumerations 1., 2., ..., 10.
    ordinalList = [
	u'',
	u'first',
	u'second',
	u'third',
	u'fourth',
	u'fifth',
	u'sixth',
	u'seventh',
	u'eighth',
	u'ninth',
	u'tenth',
	u'eleventh',
	u'twelfth',
	u'thirteenth',
	u'fourteenth',
	u'fifteenth',
	u'sixteenth',
	u'seventeenth',
	u'eighteenth',
	u'nineteenth'
	]

    # conversion of year dates--------------------------------------------------
    # example: 1978 -> 19 78
    _convertYearsRE = re.compile(r'(?P<century>19)(?P<year>\d{2})')

    def _convertYearsSub(self, m):
	return m.group('century') + ' ' + m.group('year')

    def convertYears(self, s):
	s = ' ' + s.strip() + ' '
	s = self._convertYearsRE.sub(self._convertYearsSub, s)
	return s


    # o/O in numbers -----------------------------------------------------------
    # example: 15oo gmt, 15:oo,
    _oToZeroRE = re.compile( \
	 r'(?:(?<=\d)[oO]+(?=\D?\d|[.,\s]))|' \
	 r'(?:(?<=\d\D)[oO]+(?=[\d.,\s]))|' \
	 r'(?:(?<=\d)[oO]{2,})|' \
	 r'(?:(?<=\d\D)[oO]{2,})')

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
    # example:  2,345,456
    _long1RE = re.compile(r'(?<=\D)' \
			 r'\d{1,3}(?:[ ,]\d{3})+' \
			 r'(?=\D)')

    def _long1Sub(self, m):
	return m.group(0).replace(',', '').replace(' ', '')

    def eliminateSeperators(self, s):
	s = ' ' + s.strip() + ' '
	s = self._long1RE.sub(self._long1Sub, s)
	return s


    # formatted time strings ---------------------------------------------------
    # example: 10.30 a.m.
    _timeRE = re.compile(
	r'(?P<hour>[012]?[0-9])\.(?P<minute>[0-5][0-9])'
	r'(?=\s*[ap]\.m\.|\s*noon)')
    def _makeTime(self, m):
	if m.group('minute') == '00':
	    return m.group('hour')
	else:
	    return m.group('hour') + ' ' + m.group('minute')

    def convertTimeData(self, s):
	s = ' ' + s.strip() + ' '
	s = self._timeRE.sub  (self._makeTime, s)
	return s

    # formatted date strings ---------------------------------------------------
    # example: 1: 30/11  12-11
    #          2: 26/11/76, 24-12-1998
    #          3: 16 Nov 95   7 May   11 September 2001   11 sep 01
    # -> the sixteenth of November ninety five, the 7th of May, etc.
    _date1RE = re.compile( \
	r'(?<=\s)' \
	r'(?P<day>\d{1,2})(?P<sep>[\-/])(?P<month>\d{1,2})' \
	r'(?=\s|\D\D)')
    _date2RE = re.compile( \
	r'(?<=\s)' \
	r'(?P<day>\d{1,2})(?P<sep>[\-/])(?P<month>\d{1,2})(?P=sep)(?P<year>\d{2}(?:\d{2})?)' \
	r'(?=\s|\D\D)')
    monthStrings = monthNames + \
		   [ m.lower()     for m in monthNames ] + \
		   [ m[:3]         for m in monthNames ] + \
		   [ m[:3].lower() for m in monthNames ]
    _date3RE = re.compile( \
	r'(?<=\s)' \
	r'(?P<day>\d{1,2})\s*(?P<month>' + '|'.join(monthStrings) + ')\s*(?P<year>\d{2}|\d{4})?' \
	r'(?=\s|\D)' )

    def _makeDate1(self, m):
	month = int(m.group('month'))
	day = int(m.group('day'))
	if month < 1 or month > 12 or day < 1 or day > 31:
	    return m.group(0)
	date = ' the ' + self._makeNumber(day, ordinal=True) + ' of ' + self.monthList[month]  + ' '
	return date

    def _makeDate2(self, m):
	date = self._makeDate1(m)
	year = m.group('year')
	if len(year) == 4 and year[:2] == '19':
	    date = date + ' ' + year[:2] + ' ' + year[2:] + ' '
	else:
	    date = date + ' ' + year + ' '
	return date

    def _makeDate3(self, m):
	day = int(m.group('day'))
	if day < 1 or day > 31:
	    return m.group(0)
	month = m.group('month')[:3].lower()
	month = self.monthDict[month]
	date = ' the ' + self._makeNumber(day, ordinal=True) + ' of ' + month + ' '
	year = m.group('year')
	if year:
	    if len(year) == 4 and year[:2] == '19':
		date = date + ' ' + year[:2] + ' ' + year[2:] + ' '
	    else:
		date = date + ' ' + year + ' '
	return date

    def convertDateSpecification(self, s):
	s = ' ' + s.strip() + ' '
	s = self._date1RE.sub(self._makeDate1, s)
	s = self._date2RE.sub(self._makeDate2, s)
	s = self._date3RE.sub(self._makeDate3, s)
	return s


    # formatted currencies -----------------------------------------------------
    # $100, 12,34 USD
    currencySymbols = map(re.escape, currencyDictS.keys())
    _currency1RE = re.compile( \
	r'\W' \
	r'(?P<currency>' + '|'.join(currencySymbols) + ')\s?(?P<high>\d+)(?:[.,](?P<low>\d+))?' \
	r'\s?(?P<power>million|billion|bn|m)?\s?')
    _currency2RE = re.compile( \
	r'(?P<high>\d+)(?:[.,](?P<low>\d+))?\s?(?P<power>billion|million|bn|m)?\s?(?P<currency>' + '|'.join(currencySymbols) + ')' \
	r'(?=\W|\d)' )

    def _makeCurrency(self, m):
	high = int(m.group('high'))
	power = m.group('power')
	low  = 0
	if (m.group('low') != None):
	    low = int(m.group('low'))
	currency = ''


	if (high > 0) and (low > 0):
	    currency = ' ' + m.group('high') + '.' + m.group('low') + ' '
	    if power != None:
		currency += ' ' + self.currencyAbbr[m.group('power')] + ' '
	    currency += ' ' + self.currencyDictP[m.group('currency')] + ' '

	elif (high > 0) or (low == 0):
	    currency = ' ' + m.group('high') + ' '
	    if power != None:
		if low == 0:
		    currency += ' ' + self.currencyAbbr[m.group('power')] + ' '
	    if high == 1 and low == 0:
		currency += ' ' + self.currencyDictS[m.group('currency')] + ' '
	    elif low == 0:
		currency += ' ' + self.currencyDictP[m.group('currency')] + ' '
	    if low > 0:
		if len(m.group('low')) == 1:
		    currency += ' ' + m.group('low') + '0'
		    if power != None:
			currency += ' ' + self.currencyAbbr[m.group('power')] + ' '
		    currency += ' ' + self.currencyDictP[m.group('currency')] + ' '
		else:
		    currency += ' ' + m.group('low')
		    if power != None:
			currency += ' ' + self.currencyAbbr[m.group('power')] + ' '
		    currency += ' ' + self.currencyDictP[m.group('currency')] + ' '
	else:
	    currency = ' ' + m.group('low') + ' '
	    if power != None:
		currency += ' ' + self.currencyAbbr[m.group('power')] + ' '
	    if low == 1:
		currency += ' ' + self.currencyDictS['cent'] + ' '
	    else:
		currency += ' ' + self.currencyDictP['cent'] + ' '
	return currency

    def convertCurrency(self, s):
	s = ' ' + s.strip() + ' '
	s = self._currency1RE.sub(self._makeCurrency, s)
	s = self._currency2RE.sub(self._makeCurrency, s)
	return s


    def __call__(self, s):
	s = ' '.join(s)
	s = self.convertCurrency(s)
	return s.split()


    # enumerations -------------------------------------------------------------
    # example:   <s>  3. whatever  (Only at beginning of sentence.)
    _enumerateRE = re.compile( \
	r'(?<=<s>\s)' \
	r'(\d+)\.' \
	r'(?=\D)' )

    def _makeEnumerate(self, m):
	enumerate = ''
	i = int(m.group(1))
	if i < len(self.ordinalList):
	    enumerate = self.ordinalList[i] + ' '
	else:
	    enumerate = m.group(1) + '.'
	return enumerate


    def convertEnumeration(self, s):
	s = ' ' + s.strip() + ' '
	s = self._enumerateRE.sub(self._makeEnumerate, s)
	return s


    # floats -------------------------------------------------------------------
    _floatRE = re.compile( \
	r'(?<=[^.\d])' \
	r'(\d+)\.(\d+)' \
	r'(?=\.\D|[^.\d])' )

    def _makeFloat(self, m):
	result = m.group(1) + ' point'
	for i in m.group(2):
	    result += ' ' + i
	return result

    def convertFloat(self, s):
	s = ' ' + s.strip() + ' '
	s = self._floatRE.sub(self._makeFloat, s)
	return s


    # numbers (serials, ids, versions, etc.) -----------------------------------
    # example: 2.2.1, 2.95.4-3
    _otherRE = re.compile(r'(?:\d+[.,\-/:])+\d+')

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
	r'(?=\D)')
    # example: 70s
    _intPluralRE = re.compile( \
	r'(?<=\D)' \
	r'(?P<number>\d+)s' \
	r'(?=\W)' )
    # example: 1st 2nd 3rd 21st 5th
    _intOrdinalRE = re.compile( \
	r'(?<=\D)' \
	r'(?P<number>\d*(1st|2nd|3rd|11th|12th|13th|[0456789]th))' \
	r'(?=\W)' )

    def _makeNumber(self, i, ordinal=False):
	number = ''
	if i < 20:
	    if ordinal:
		number += self.ordinalList[i]
	    else:
		number += self.numberList[i]
	elif i < 100:
	    i_div = i / 10
	    i_mod = i % 10
	    number = self.decList[i_div]
	    if i_mod > 0:
		if ordinal:
		    number += ' ' + self.ordinalList[i_mod]
		else:
		    number += ' ' + self.numberList[i_mod]
	    elif ordinal:
		number = number[:-1] + 'ieth'
	elif i < 1000:
	    i_div = i / 100
	    i_mod = i % 100
	    number = self.numberList[i_div] + ' ' + self.powerList[2]
	    if i_mod > 0:
		number += ' ' + self._makeNumber(i_mod, ordinal)
	    elif ordinal:
		number += 'th'
	elif i < 1000000:
	    i_div = i / 1000
	    i_mod = i % 1000
	    number = self._makeNumber(i_div) + ' ' + self.powerList[3]
	    if i_mod > 0:
		number += ' ' + self._makeNumber(i_mod, ordinal)
	    elif ordinal:
		number += 'th'
	else:
	    i_div = i / 1000000
	    i_mod = i % 1000000
	    number = self._makeNumber(i_div) + ' ' + self.powerList[6]
	    if i_mod > 0:
		number += ' ' + self._makeNumber(i_mod, ordinal)
	    elif ordinal:
		number += 'th'
	return number

    def _makeInt(self, m):
	number = m.group('number')
	length = len(number)
	suffix = ' '

	while (length >= 10):
	    suffix += self.powerList[6] + ' ' + self._makeNumber(int(number[-6:])) + ' '
	    number = number[:-6]
	    length -= 6

	str = self._makeNumber(int(number))
	return ' ' + str + suffix

    def _makeIntPlural(self, m):
	s = self._makeInt(m).strip()
	if s.endswith('y'):
	    s = s[:-1] + 'ies'
	else:
	    s = s + 's'
	return ' ' + s + ' '

    def _makeIntOrdinal(self, m):
	number = m.group('number')
	s = self._makeNumber(int(number[:-2]), ordinal=True)
	return ' ' + s + ' '

    def convertNumber(self, s):
	s = ' ' + s.strip() + ' '
	s = self._intOrdinalRE.sub(self._makeIntOrdinal, s)
	s = self._intPluralRE.sub(self._makeIntPlural, s)
	s = self._intRE.sub(self._makeInt, s)
	return s


    # --------------------------------------------------------------------------
    def __call__(self, s):
	s = ' '.join(s)
	s = self.convertYears(s)
	s = self.convertO(s)
	s = self.eliminateSeperators(s)
	s = self.convertTimeData(s)
	s = self.convertDateSpecification(s)
	s = self.convertCurrency(s)
	s = self.convertEnumeration(s)
	s = self.convertFloat(s)
	s = self.convertOther(s)
	s = self.convertRoman(s)
	s = self.convertNumber(s)
	return s.split()


# ===========================================================================

letters = [
    (line.split()[0], tuple(line.split()[1:]))
    for line in u"""\
	A       " ey
	B       " b iy
	C       " s iy
	D       " d iy
	E       " iy
	F       " eh f
	G       " jh iy
	H       " ey ch
	I       " ay
	J       " jh ax iy
	K       " k ey
	L       " eh l
	M       " eh m
	N       " eh n
	O       " ow
	P       " p iy
	Q       " k y uw
	R       " aa r
	S       " eh s
	T       " t iy
	U       " y uw
	V       " v iy
	W       " d ah . b l . y uw
	X       " eh k s
	Y       " w ay
	Z       " z eh d                     """.split('\n') ]
