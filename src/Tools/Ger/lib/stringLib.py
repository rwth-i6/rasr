# -*- coding: iso-8859-1 -*-

def isMasked(s, i):
    """
    s is a string and i a valid index for s, i.e. 0 <= i < len(s)
    Returns true, if s[i] is preceeded by an odd number of '\'
    """
    if i == 0:
	return False
    if not s[i-1] == '\\':
	return False
    masked == True
    i -= 2
    while i >= 0 and s[i] == '\\':
	masked = not masked
	i -= 1
    return masked

def find_unmasked(s, c, b = 0):
    """
    Finds index of first occurence of character c in string s starting from index b,
    such that c is not masked, i.e. not preceeded by on odd number of '\'.
    If c is not found, -1 is returned
    """
    while True:
	e = s.find(c, b)
	if e == -1:
	    return -1
	elif not isMasked(s, e):
	    return e
	else:
	    b = e + 1

def next_in(s, c, b, e):
    """
    Finds index i of first occurence of character c in string s starting from index b,
    such that i is less then e.
    If c is not found, e is returned.
    c might be a string and is handled as a set of characters.
    """
    while b < e and s[b] not in c:
	b += 1
    return b

def next_not_in(s, c, b, e):
    """
    Finds index i of first non-occurence of character c in string s starting from index b,
    such that i is less then e.
    If c is not found, e is returned.
    c might be a string and is handled as a set of characters.
    """
    while b < e and s[b] in c:
	b += 1
    return b
