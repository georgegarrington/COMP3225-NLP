﻿# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2021
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2021/01/29
# Project : Teaching
#
######################################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, re, logging
warnings.simplefilter( action='ignore', category=FutureWarning )

import nltk, numpy, scipy, sklearn, sklearn_crfsuite, sklearn_crfsuite.metrics

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')

def exec_ner( file_chapter = None, ontonotes_file = None ) :

	# INSERT CODE TO TRAIN A CRF NER MODEL TO TAG THE CHAPTER OF TEXT (subtask 3)
	# USING NER MODEL AND REGEX GENERATE A SET OF BOOK CHARACTERS AND FILTERED SET OF NE TAGS (subtask 4)

	# hardcoded output to show exactly what is expected to be serialized (you should change this)
	dictNE = {
			"CARDINAL": [
				"two",
				"three",
				"one"
			],
			"ORDINAL": [
				"first"
			],
			"DATE": [
				"saturday",
			],
			"NORP": [
				"indians"
			],
			"PERSON": [
				"creakle",
				"mr. creakle",
				"mrs. creakle",
				"miss creakle"
			]
		}

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	# write out all PERSON entries for character list for subtask 4
	writeHandle = codecs.open( 'characters.txt', 'w', 'utf-8', errors = 'replace' )
	if 'PERSON' in dictNE :
		for strNE in dictNE['PERSON'] :
			writeHandle.write( strNE.strip().lower()+ '\n' )
	writeHandle.close()

	# FILTER NE dict by types required for subtask 3
	listAllowedTypes = [ 'DATE', 'CARDINAL', 'ORDINAL', 'NORP' ]
	listKeys = list( dictNE.keys() )
	for strKey in listKeys :
		for nIndex in range(len(dictNE[strKey])) :
			dictNE[strKey][nIndex] = dictNE[strKey][nIndex].strip().lower()
		if not strKey in listAllowedTypes :
			del dictNE[strKey]

	# write filtered NE dict
	writeHandle = codecs.open( 'ne.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictNE, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

def exec_regex_toc( file_book = None ) :

	# INSERT CODE TO USE REGEX TO BUILD A TABLE OF CONTENTS FOR A BOOK (subtask 1)

	# hardcoded output to show exactly what is expected to be serialized
	dictTOC = {
			"1": "I AM BORN",
			"2": "I OBSERVE",
			"3": "I HAVE A CHANGE"
		}

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	writeHandle = codecs.open( 'toc.json', 'w', 'utf-8', errors = 'replace' )
	strJSON = json.dumps( dictTOC, indent=2 )
	writeHandle.write( strJSON + '\n' )
	writeHandle.close()

def exec_regex_questions( file_chapter = None ) :

	# INSERT CODE TO USE REGEX TO LIST ALL QUESTIONS IN THE CHAPTER OF TEXT (subtask 2)

	# hardcoded output to show exactly what is expected to be serialized
	setQuestions = set([
		"Traddles?",
		"And another shilling or so in biscuits, and another in fruit, eh?",
		"Perhaps you’d like to spend a couple of shillings or so, in a bottle of currant wine by and by, up in the bedroom?",
		"Has that fellow’--to the man with the wooden leg--‘been here again?"
		])

	# DO NOT CHANGE THE BELOW CODE WHICH WILL SERIALIZE THE ANSWERS FOR THE AUTOMATED TEST HARNESS TO LOAD AND MARK

	writeHandle = codecs.open( 'questions.txt', 'w', 'utf-8', errors = 'replace' )
	for strQuestion in setQuestions :
		writeHandle.write( strQuestion + '\n' )
	writeHandle.close()

if __name__ == '__main__':
	if len(sys.argv) < 4 :
		raise Exception( 'missing command line args : ' + repr(sys.argv) )
	ontonotes_file = sys.argv[1]
	book_file = sys.argv[2]
	chapter_file = sys.argv[3]

	logger.info( 'ontonotes = ' + repr(ontonotes_file) )
	logger.info( 'book = ' + repr(book_file) )
	logger.info( 'chapter = ' + repr(chapter_file) )

	# DO NOT CHANGE THE CODE IN THIS FUNCTION

	#
	# subtask 1 >> extract chapter headings and create a table of contents from a provided plain text book (from www.gutenberg.org)
	# Input >> www.gutenberg.org sourced plain text file for a whole book
	# Output >> toc.json = { <chapter_number_text> : <chapter_title_text> }
	#

	exec_regex_toc( book_file )

	#
	# subtask 2 >> extract every question from a provided plain text chapter of text
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> questions.txt = plain text set of extracted questions. one line per question.
	#

	exec_regex_questions( chapter_file )

	#
	# subtask 3 (NER) >> train NER using ontonotes dataset, then extract DATE, CARDINAL, ORDINAL, NORP entities from a provided chapter of text
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> ne.json = { <ne_type> : [ <phrase>, <phrase>, ... ] }
	#
	# subtask 4 (text classifier) >> compile a list of characters from the target chapter
	# Input >> www.gutenberg.org sourced plain text file for a chapter of a book
	# Output >> characters.txt = plain text set of extracted character names. one line per character name.
	#

	exec_ner( chapter_file, ontonotes_file )

