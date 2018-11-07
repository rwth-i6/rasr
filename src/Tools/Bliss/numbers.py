# *****************************************************************************
# parsing of numbers; possible language codes:
# en_EN_simple: english unhyphenated (default)
# en_EN       : english with hyphens
# de_DE       : german
def numberToText( number, languageCode="en_EN_simple" ):
    NUMBERMAP_en_EN = {	'0'	:	'oh',
			'1'	:	'one',
			'2'	:	'two',
			'3'	:	'three',
			'4'	:	'four',
			'5'	:	'five',
			'6'	:	'six',
			'7'	:	'seven',
			'8'	:	'eight',
			'9'	:	'nine',
			'10'	:	'ten',
			'11'	:	'eleven',
			'12'	:	'twelve',
			'13'	:	'thirteen',
			'14'	:	'fourteen',
			'15'	:	'fifteen',
			'16'	:	'sixteen',
			'17'	:	'seventeen',
			'18'	:	'eighteen',
			'19'	:	'nineteen',
			'20'	:	'twenty',
			'30'	:	'thirty',
			'40'	:	'forty',
			'50'	:	'fifty',
			'60'	:	'sixty',
			'70'	:	'seventy',
			'80'	:	'eighty',
			'90'	:	'ninety'
			}
    NUMBERMAP_de_DE = { '0'	:	'null',
			'1'	:	'eins',
			'2'	:	'zwei',
			'3'	:	'drei',
			'4'	:	'vier',
			'5'	:	'fuenf',
			'6'	:	'sechs',
			'7'	:	'sieben',
			'8'	:	'acht',
			'9'	:	'neun',
			'10'	:	'zehn',
			'11'	:	'elf',
			'12'	:	'zwoelf',
			'13'	:	'dreizehn',
			'14'	:	'vierzehn',
			'15'	:	'fuenfzehn',
			'16'	:	'sechzehn',
			'17'	:	'siebzehn',
			'18'	:	'achtzehn',
			'19'	:	'neunzehn',
			'20'	:	'zwanzig',
			'30'	:	'dreissig',
			'40'	:	'vierzig',
			'50'	:	'fuenfzig',
			'60'	:	'sechzig',
			'70'	:	'siebzig',
			'80'	:	'achtzig',
			'90'	:	'neunzig'
			}
    NUMBERMAP_es_ES = {'0'      :       'cero',
		       '1'      :       'un/o',
		       '2'      :       'dos',
		       '3'      :       'tres',
		       '4'      :       'cuatro',
		       '5'      :       'cinco',
		       '6'      :       'seis',
		       '7'      :       'siete',
		       '8'      :       'ocho',
		       '9'      :       'nueve',
		       '10'     :       'diez',
		       '11'     :       'once',
		       '12'     :       'doce',
		       '13'     :       'trece',
		       '14'     :       'catorce',
		       '15'     :       'quince',
		       '16'     :       'dieciséis',
		       '17'     :       'diecisiete',
		       '18'     :       'dieciocho',
		       '19'     :       'diecinueve',
		       '20'     :       'veinte',
		       '21'     :       'veintiuno',
		       '22'     :       'veintidós',
		       '23'     :       'veintitrés',
		       '24'     :       'veinticuatro',
		       '25'     :       'veinticinco',
		       '26'     :       'veintiséis',
		       '27'     :       'veintisiete',
		       '28'     :       'veintiocho',
		       '29'     :       'veintinueve',
		       '30'     :       'treinta',
		       '40'     :       'cuarenta',
		       '50'     :       'cincuenta',
		       '60'     :       'sesenta',
		       '70'     :       'setenta',
		       '80'     :       'ochenta',
		       '90'     :       'noventa'
		       }
    HUNDREDS_es_ES = {'1'      :       'ciento',
		      '2'      :       'doscientos/as',
		      '3'      :       'trescientos/as',
		      '4'      :       'cuatrocientos/as',
		      '5'      :       'quinientos/as',
		      '6'      :       'seiscientos/as',
		      '7'      :       'setecientos/as',
		      '8'      :       'ochocientos/as',
		      '9'      :       'novecientos/as'
		      }

    if len(number) > 1 and number[0] == '0':
	return(number)

    if languageCode == "en_EN" or languageCode == "en_EN_simple":
	NUMBERMAP = NUMBERMAP_en_EN
	if NUMBERMAP.has_key(number):
	    return (NUMBERMAP[number])
	else:
	    if len(number) == 2:
		main = number[:-1]+'0'
		rest = re.sub('^0+', '', number[1:])
		if languageCode == "en_EN":
		    return( numberToText(main, languageCode) + '-' + numberToText(rest, languageCode))
		elif languageCode == "en_EN_simple":
		    return( numberToText(main, languageCode) + ' ' + numberToText(rest, languageCode))
	    elif len(number) == 3:
		main = number[:-2]
		rest = re.sub('^0+', '', number[1:])
		if rest == '':
		    return(numberToText(main, languageCode)+' hundred')
		else:
		    return(numberToText(main, languageCode)+' hundred '+numberToText(rest, languageCode))
	    elif len(number) > 3 and len(number) < 7:
		main = number[:-3]
		rest = re.sub('^0+', '', number[(len(number)-3):])
		if rest == '':
		    return(numberToText(main, languageCode)+' thousand')
		else:
		    return(numberToText(main, languageCode)+' thousand '+numberToText(rest, languageCode))
	    elif len(number) >= 7 and len(number) < 10:
		main = number[:-6]
		rest = re.sub('^0+', '', number[(len(number)-6):])
		if rest == '':
		    return(numberToText(main, languageCode)+' million')
		else:
		    return(numberToText(main, languageCode)+' million '+numberToText(rest, languageCode))
    elif languageCode == "de_DE":
	NUMBERMAP = NUMBERMAP_de_DE
	if NUMBERMAP.has_key(number):
	    return (NUMBERMAP[number])
	else:
	    if len(number) == 2:
		main = number[:-1]+'0'
		rest = re.sub('^0+', '', number[1:])
		if number.endswith('1'):
		    return('ein' + 'und' + numberToText(main,languageCode))
		else:
		    return(numberToText(rest, languageCode) + 'und' + numberToText(main, languageCode))
	    elif len(number) == 3:
		main = number[:-2]
		rest = re.sub('^0+', '', number[1:])
		if rest == '':
		    if main.startswith('1') and len(number)==3:
			return('ein hundert')
		    else:
			return(numberToText(main, languageCode)+' hundert')
		else:
		    if main.startswith('1') and len(number)==3:
			return('ein hundert ' +numberToText(rest, languageCode))
		    else:
			return(numberToText(main, languageCode)+' hundert '+numberToText(rest, languageCode))
	    elif len(number) > 3 and len(number) < 7:
		main = number[:-3]
		rest = re.sub('^0+', '', number[(len(number)-3):])
		if rest == '':
		    if main.startswith('1') and len(number)==4:
			return('ein tausend')
		    else:
			return(numberToText(main, languageCode)+' tausend')
		else:
		    if main.startswith('1') and len(number)==4:
			return('ein tausend ' +numberToText(rest, languageCode))
		    else:
			return(numberToText(main, languageCode)+' tausend '+numberToText(rest, languageCode))
	    elif len(number) >= 7 and len(number) < 10:
		main = number[:-6]
		rest = re.sub('^0+', '', number[(len(number)-6):])
		if rest == '':
		    if main.startswith('1') and len(number)==7:
			return('eine million')
		    else:
			return(numberToText(main, languageCode)+' millionen')
		else:
		    if main.startswith('1') and len(number)==7:
			return('eine million ' +numberToText(rest, languageCode))
		    else:
			return(numberToText(main, languageCode)+' millionen '+numberToText(rest, languageCode))
    elif languageCode == "es_ES":
	NUMBERMAP = NUMBERMAP_es_ES
	if NUMBERMAP.has_key(number):
	    return(NUMBERMAP[number])
	else:
	    if len(number) == 2:
		main = number[:-1]+'0'
		rest = re.sub('^0+', '', number[1:])
		return(numberToText(main, languageCode) + ' y ' + numberToText(rest, languageCode))
	    elif len(number) == 3:
		main = number[:-2]
		rest = re.sub('^0+', '', number[1:])
		if rest == '':
		    if main.startswith('1') and len(number)==3:
			try:
			    test_cien
			except NameError:
			    test_cien = True
			    return('cien')
			else:
			    if test_cien == True:
				return('cien')
			    else:
				return('ciento')
		    else:
			return(HUNDREDS_es_ES[main])
		else:
		    return(HUNDREDS_es_ES[main] + ' ' + numberToText(rest, languageCode))
	    elif len(number) > 3 and len(number) < 7:
		test_cien = False
		main = number[:-3]
		rest = re.sub('^0+', '', number[(len(number)-3):])
		if rest == '':
		    if main.startswith('1') and len(number)==4:
			return('mil')
		    else:
			return(numberToText(main, languageCode)+' mil')
		else:
		    if main.startswith('1') and len(number)==4:
			return('mil ' +numberToText(rest, languageCode))
		    else:
			return(numberToText(main, languageCode)+' mil '+numberToText(rest, languageCode))
	    elif len(number) >= 7 and len(number) < 10:
		test_cien = False
		main = number[:-6]
		rest = re.sub('^0+', '', number[(len(number)-6):])
		if rest == '':
		    if main.startswith('1') and len(number)==7:
			return('un millón')
		    else:
			return(numberToText(main, languageCode)+' milliones')
		else:
		    if main.startswith('1') and len(number)==7:
			return('un millión ' +numberToText(rest, languageCode))
		    else:
			return(numberToText(main, languageCode)+' milliones '+numberToText(rest, languageCode))

    return( number )

# ****************************************************************************

def convertRomanToArabic( romanNumber ):

    romanNumber  = romanNumber.upper()
    arabicNumber = 0

    ROMANNUMBERMAP= ( ('M',  1000, re.compile('^M{1,3}')),
		      ('CM', 900,  re.compile('^CM')),
		      ('D',  500,  re.compile('^D')),
		      ('CD', 400,  re.compile('^CD')),
		      ('C',  100,  re.compile('^C{1,3}')),
		      ('XC', 90,   re.compile('^XC')),
		      ('L',  50,   re.compile('^L')),
		      ('XL', 40,   re.compile('^XL')),
		      ('X',  10,   re.compile('^X{1,3}')),
		      ('IX', 9,    re.compile('^IX')),
		      ('V',  5,    re.compile('^V')),
		      ('IV', 4,    re.compile('^IV')),
		      ('I',  1,    re.compile('^I{1,3}'))
		    )
    for roman, arabic, pattern in ROMANNUMBERMAP:
	match = pattern.search(romanNumber)
	if match:
	    lenMatch = len(match.group())
	    arabicNumber += arabic * lenMatch / len(roman)
	    romanNumber = romanNumber[lenMatch:]

    return( arabicNumber )

# ****************************************************************************

def convertNumbers(sentenceString):
     sentenceString = re.sub(r'(\()([0-9]+)', r'\1 \2', sentenceString)
     sentenceString = re.sub(r'([0-9]) +(000) ', r'\1\2 ', sentenceString)
     sentenceString = re.sub(r'([\/])([0-9]+)', r'\1 \2', sentenceString)
     sentenceString = re.sub(r'([0-9]+)([\/])', r'\1 \2', sentenceString)
     sentenceString = re.sub(r'([0-9]+)([a-z]+)', r'\1 \2', sentenceString)
     wordList = sentenceString.split(' ')
     for word in range(len(wordList)):
	  if re.compile('[0-9]+').search(wordList[word]) != None:
	       # two numbers with hyphen
	       if re.compile('([0-9]+(,|\.){0,1}[0-9]*)(\-)([0-9]+(,|\.){0,1}[0-9]*)').match(wordList[word]):
		    tempList = wordList[word].split('-')
		    wordList[word] = convertNumbers(tempList[0]) + ' till ' + convertNumbers(tempList[1])
	       # number with percentage
	       elif re.compile('[0-9]{0,3}(\.[0-9]{0,2}){0,1}%').match(wordList[word]):
		    tempNumber = wordList[word].split('%')[0]
		    if tempNumber.find('.') == -1:
			 wordList[word] = numberToText(tempNumber) + ' percent'
		    else:
			 tempList = tempNumber.split('.')
			 tempNumber = ''
			 for i in range(len(tempList[1])):
			      tempNumber += ' ' + numberToText(tempList[1][i])
			 wordList[word] = numberToText(tempList[0])+ ' point' + tempNumber + ' percent'
	       # number containing a colon
	       elif re.compile('[0-9]+,[0-9]+').match(wordList[word]):
		    tempList = wordList[word].split(',')
		    tempNumber = ''
		    for i in range(len(tempList[1])):
			 tempNumber += ' ' + numberToText(tempList[1][i])
		    wordList[word] = numberToText(tempList[0])+ ' point' + tempNumber
	       # number containing a point
	       elif re.compile('[0-9]+\.[0-9]+').match(wordList[word]):
		    tempList = wordList[word].split('.')
		    tempNumber = ''
		    for i in range(len(tempList[1])):
			 tempNumber += ' ' + numberToText(tempList[1][i])
		    wordList[word] = numberToText(tempList[0])+ ' point' + tempNumber
	       # years
	       elif re.compile('19[0-9]{2}').match(wordList[word]):
		   if len(re.sub('19[0-9]{2}', '', wordList[word])) == 0:
		       number = wordList[word]
		       main = number[:-2]
		       rest = re.sub('^0+', '', number[2:])
		       if rest == '':
			   wordList[word] = numberToText(main) + ' hundred'
		       else:
			   wordList[word] = numberToText(main) + ' ' +numberToText(rest)
		   elif len(re.sub('19[0-9]{2}([^0-9]+(\.|:|;|,|!|\?){0,1})', '', wordList[word]))== 0:
		       boundary = wordList[word].find(re.compile('([a-z]+)|:|;|,|!|\?|\.|-|\(|\)|\'|\+').search(wordList[word]).group())
		       number = wordList[word][:boundary]
		       main = number[:-2]
		       rest = re.sub('^0+', '', number[2:])
		       if rest == '':
			   wordList[word] = numberToText(main) + ' hundred' + wordList[word][boundary:]
		       else:
			   wordList[word] = numberToText(main) + ' ' +numberToText(rest) + wordList[word][boundary:]
	       # arbitrarily big numbers, also occuring bevor punctuation marks
	       elif re.compile('[1-9][0-9]*(:|;|,|!|\?){0,1}([^0-9]*(:|;|,|!|\?){0,1}|[^0-9]+(\.){1})').match(wordList[word]):
		    if len(re.sub('[1-9][0-9]*', '', wordList[word])) == 0:
			 wordList[word] = numberToText(wordList[word])
		    elif len(re.sub('[1-9][0-9]*(:|;|,|!|\?){1}', '', wordList[word])) == 0:
			 wordList[word] = numberToText(wordList[word][:-1]) + wordList[word][-1]
		    elif len(re.sub('[1-9][0-9]*([^0-9]+(\.|:|;|,|!|\?){0,1})', '', wordList[word]))== 0:
			 boundary = wordList[word].find(re.compile('([A-Z|a-z]+)|:|;|,|!|\?|\.|-|\(|\)|\'|\+').search(wordList[word]).group())
			 # ordinals
			 if len(re.sub('[1-9][0-9]*(st|nd|rd|th)', '', wordList[word])) == 0:
			     wordList[word] = getOrdinal(wordList[word][:boundary])
			 else:
			     wordList[word] = numberToText(wordList[word][:boundary]) + wordList[word][boundary:]
	       # roman numbers
	       #elif re.compile('(I|V|X|L|C|D|M)+').match(wordList[word]):
	       #     if len(re.sub('(I|V|X|L|C|D|M)+', '', wordList[word])) == 0:
	       #          tempNumber = str(convertRomanToArabic(wordList[word]))
	       #          dataList[word] = numberToText(tempNumber)
	       #     elif len(re.sub('(I|V|X|L|C|D|M)+(\.|-|:|;|,|!|\?){0,1}', '', wordList[word])) == 0:
	       #          tempNumber = str(convertRomanToArabic(wordList[word][:-1]))
	       #          wordList[word] = numberToText(tempNumber) + wordList[word][-1]
	       #     elif len(re.sub('(I|V|X|L|C|D|M)+(-){1}', '', wordList[word])) == len(wordList[word][wordList[word].find('-'):])-1:
	       #          tempNumber = str(convertRomanToArabic(wordList[word].split('-')[0]))
	       #          wordList[word] = numberToText(tempNumber) +  wordList[word][wordList[word].find('-'):]

     sentence = string.join(wordList)
     return(sentence)

# ****************************************************************************

def getOrdinal(number):
    if number == '1':
	ordinal = 'first'
    elif number == '2':
	ordinal = 'second'
    elif number == '3':
	ordinal = 'third'
    elif number == '5':
	ordinal = 'fifth'
    elif number == '8':
	ordinal = 'eighth'
    elif number == '12':
	ordinal = 'twelfth'
    elif int(number) < 20:
	ordinal = numberToText(number) + 'th'
    elif (int(number) > 19) and (int(number) < 100) and (int(number) % 10 == 0):
	ordinal = numberToText(number)[:-1] + 'ieth'
    elif (int(number) > 20) and (int(number) < 100) and (int(number) % 10 != 0):
	ordinal = numberToText(number[:-1]+'0') + '-' + getOrdinal(number[-1])
    elif (int(number) > 99) and (int(number) < 1000):
	if (int(number) % 100 == 0):
	    ordinal = numberToText(number) + 'th'
	else:
	    ordinal = numberToText(number[0]+'00') +' '+ getOrdinal(number[1:])
    return(ordinal)

# ****************************************************************************

def convertDates(sentenceString):
    wordList = sentenceString.split(' ')
    months = re.compile('(January|February|March|April|May|June|July|August|September|October|November|December)')
    for word in range(len(wordList)):
	if len(wordList[word]) > 0 and len(re.sub('[0-3]*[0-9]', '', wordList[word])) == 0 and len(wordList)-1 > word:
	    if months.match(wordList[word+1]):
		wordList[word] = getOrdinal(wordList[word]) + ' of'
    sentence = string.join(wordList)
    return(sentence)

# ===========================================================================

currencyList = [
    ('EUR',     'euros'),
    (u'\u20ac', 'euros'),
    ('ECU',     'ECU'),
    ('USD',     'US dollars'),
    ('US$',     'US dollars'),
    ('$',       'dollars'),
    ('pound',   'pounds'),
    ( u'\xa3',  'pounds')
    ]

reCurrency = re.compile(
    '(' + '|'.join([ re.escape(symbol) for symbol, text in currencyList ]) + ')')

def convertCurrency(sentenceString):
    sentenceString = re.sub(r'([0-9]+[,|\.]{0,1}[0-9]+)(\. |, |:|;|!|\?)', r'\1 \2', sentenceString)
    sentenceString = re.sub(r'(million|billion)(\. |, |:|;|!|\?)', r'\1 \2', sentenceString)
    sentenceString = re.sub(r'( m| bn)(\. |, |:|;|!|\?)', r'\1 \2', sentenceString)
    wordList = sentenceString.split(' ')
    for word in range(len(wordList)):
	if re.compile('[0-9]+').search(wordList[word]) != None:
	    if re.compile('[1-9][0-9]*').match(wordList[word]) and reCurrency.match(wordList[word-1]):
		currency = wordList[word-1]
		for symbol, text in currencyList:
		    currency = currency.replace(symbol, text)
		amountAbbr = re.compile('(m|bn)')
		amount = re.compile('(million|billion)')
		if len(amountAbbr.sub('', wordList[word+1]))==0:
		    amountList = [
			['bn', 'billion'],
			['m', 'million']
			]
		    for pair in amountList:
			wordList[word+1] = wordList[word+1].replace(pair[0], pair[1])
		# if the amount is a floating point number
		if re.compile('[0-9]+[\.]{1}[0-9]+').match(wordList[word]):
		    tempList = wordList[word].split('.')
		    tempNumber = ''
		    for i in range(len(tempList[1])):
			tempNumber += ' ' + numberToText(tempList[1][i])
			wordList[word-1] = numberToText(tempList[0]) + ' point' + tempNumber
		    if amount.match(wordList[word+1]):
			wordList[word] = wordList[word+1]
			wordList[word+1] = currency
		# if the amount has dangling zeros, separated by ' '
		elif re.compile('[000]{1}').match(wordList[word+1]):
		    tempNumber = wordList[word] + wordList[word+1]
		    wordList[word+1] = '\b'
		    wordList[word-1] = numberToText(tempNumber)
		    wordList[word] = currency
		else:
		    wordList[word-1] = numberToText(wordList[word])
		    if amount.match(wordList[word+1]):
			wordList[word] = wordList[word+1]
			wordList[word+1] = currency
		    else:
			wordList[word] = currency
    sentence = string.join(wordList)
    return(sentence)
