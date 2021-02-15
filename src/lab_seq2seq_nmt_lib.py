# !/usr/bin/env python
# -*- coding: utf-8 -*-

######################################################################
#
# (c) Copyright University of Southampton, 2020
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Stuart E. Middleton
# Created Date : 2020/08/12
# Project : Teaching
#
######################################################################
#
# Code derived from TensorFlow (Apache 2.0 license) tutorial code see https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
#
######################################################################

import unicodedata, re, io, time, os, datetime, sys, codecs, math, gc, random, contextlib, itertools, json, string
import tensorflow as tf

table_punct = str.maketrans('', '', string.punctuation)
re_print = re.compile('[^%s]' % re.escape(string.printable))

def normalize_sent_regex( sent, logger ) :
	# normalizing unicode characters
	line = unicodedata.normalize('NFD', sent).encode('ascii', 'ignore')
	line = line.decode('UTF-8')

	# tokenize on white space
	line = line.split()

	# convert to lowercase
	line = [word.lower() for word in line]

	# removing punctuation
	line = [word.translate(table_punct) for word in line]

	# removing non-printable chars form each token
	line = [re_print.sub('', w) for w in line]

	# removing tokens with numbers
	line = [word for word in line if word.isalpha()]

	# return normalized text
	return ' '.join(line)

def normalize_sent_moses( sent, logger, word_tokenizer = None, punc_normalization = None ) :
	result = sent

	try :
		list_tokens = word_tokenizer( punc_normalization( result ) )
	except :
		logger.info( 'T1 = ' + repr(result) )
		logger.info( 'T2 = ' + repr(type(result)) )
		raise

	result = ' '.join( list_tokens )

	return result

def load_corpus( filename, logger, normalize_func, **kwargs ) :
	logger.info( 'loading file ' + filename )
	read_handle = codecs.open( filename, 'r', 'utf-8', errors = 'strict' )
	str_text = read_handle.read()
	read_handle.close()

	list_lines = str_text.split( '\n' )

	list_empty = []

	nSentIndex = 0
	while nSentIndex < len(list_lines) :
		# get line
		line = list_lines[nSentIndex]
		line = line.strip().rstrip('.').replace( '\n', ' ' )

		# keep cleaned string which can be tokenized easily
		if len(list_lines[nSentIndex]) > 0 :
			list_lines[nSentIndex] = normalize_func( line, logger = logger, **kwargs )
		else :
			list_empty.append( nSentIndex )

		nSentIndex += 1

	# remove empty lines (whitespace should be mirrored in bitext so this will not change sent index alignment)
	list_empty = sorted( list_empty, reverse = True )
	for nSentIndex in list_empty :
		del list_lines[nSentIndex]

	return list_lines

def train_tokenizer( list_lines, logger ) :
	tokenizer = tf.keras.preprocessing.text.Tokenizer( filters='', lower = False, split = ' ' )
	tokenizer.fit_on_texts( list_lines )

	# remake the word_index so any word that is not alpha or single char punctuation is removed from the vocabulary
	# this means digits, symbols and alphanumerics will not appear, and so be treated as <unkpos0>
	# at translation time we will rely on the non-alpha identify mapping to copy the original non-alpha to replace the predicted <unkpos0> value
	dictNewWordIndex = {}
	dictNewIndexWord = {}
	nNewWordIndex = 1
	for strWord in tokenizer.word_index :
		if (strWord.isalpha() == True) or (strWord.startswith('&apos;') == True) or (strWord.startswith('&quot;') == True) or ((len(strWord) == 1) and (strWord in string.punctuation)) :
			dictNewWordIndex[ strWord ] = nNewWordIndex
			dictNewIndexWord[ nNewWordIndex ] = strWord
			nNewWordIndex += 1
	tokenizer.word_index = dictNewWordIndex
	tokenizer.index_word = dictNewIndexWord
	tokenizer.word_counts = {}
	tokenizer.word_docs = {}
	tokenizer.index_docs = {}

	# declare the <unk> token AFTER we call fit_on_texts() and prune it for vocab size
	tokenizer.oov_token = '<unk>'
	if not '<unk>' in tokenizer.word_index :
		nUnkWordIndex = len(tokenizer.word_index) + 1
		tokenizer.word_index['<unk>'] = nUnkWordIndex
		tokenizer.index_word[nUnkWordIndex] = '<unk>'

	# add <pad> <start> amnd <end> tokens for nmt later
	nPadWordIndex = len(tokenizer.word_index) + 1
	tokenizer.word_index['<pad>'] = nPadWordIndex
	tokenizer.index_word[nPadWordIndex] = '<pad>'

	nStartWordIndex = len(tokenizer.word_index) + 1
	tokenizer.word_index['<start>'] = nStartWordIndex
	tokenizer.index_word[nStartWordIndex] = '<start>'

	nEndWordIndex = len(tokenizer.word_index) + 1
	tokenizer.word_index['<end>'] = nEndWordIndex
	tokenizer.index_word[nEndWordIndex] = '<end>'

	return tokenizer

def train_tokenizer_top_N( list_lines, top_N, is_source, logger ) :
	tokenizer = tf.keras.preprocessing.text.Tokenizer( filters='', lower = False, split = ' ' )
	tokenizer.fit_on_texts( list_lines )

	# remake the word_index so any word that is not alpha or single char punctuation is removed from the vocabulary
	# this means digits, symbols and alphanumerics will not appear, and so be treated as <unkpos0>
	# at translation time we will rely on the non-alpha identify mapping to copy the original non-alpha to replace the predicted <unkpos0> value
	dictNewWordIndex = {}
	dictNewIndexWord = {}
	nNewWordIndex = 1
	for strWord in tokenizer.word_index :
		if (strWord.isalpha() == True) or (strWord.startswith('&apos;') == True) or (strWord.startswith('&quot;') == True) or ((len(strWord) == 1) and (strWord in string.punctuation)) :
			dictNewWordIndex[ strWord ] = nNewWordIndex
			dictNewIndexWord[ nNewWordIndex ] = strWord
			nNewWordIndex += 1
	tokenizer.word_index = dictNewWordIndex
	tokenizer.index_word = dictNewIndexWord
	tokenizer.word_counts = {}
	tokenizer.word_docs = {}
	tokenizer.index_docs = {}

	# manually truncate the word index list
	# this is not done by tf.keras.preprocessing.text.Tokenizer if we use num_tokens argument as it keeps the full word index list anyway (sorted by freq)
	# source = vocab - 4 allowing for unk, start, end, pad
	# target = vocab - 18 allowing for unkpos-7 ... unkpos7, start, end, pad
	if top_N != None :
		if is_source == True :
			tokenizer.word_index = { e:i for e,i in tokenizer.word_index.items() if i <= top_N - 4 }
		else :
			tokenizer.word_index = { e:i for e,i in tokenizer.word_index.items() if i <= top_N - 18 }

	# declare the <unk> token AFTER we call fit_on_texts() and prune it for vocab size
	if is_source == True :
		tokenizer.oov_token = '<unk>'
		if not '<unk>' in tokenizer.word_index :
			nUnkWordIndex = len(tokenizer.word_index) + 1
			tokenizer.word_index['<unk>'] = nUnkWordIndex
			tokenizer.index_word[nUnkWordIndex] = '<unk>'
	else :
		tokenizer.oov_token = '<unkpos0>'
		for nUnkPos in range( -7,8 ) :
			strUnkToken = '<unkpos' + str(nUnkPos) + '>'
			if not strUnkToken in tokenizer.word_index :
				nUnkWordIndex = len(tokenizer.word_index) + 1
				tokenizer.word_index[strUnkToken] = nUnkWordIndex
				tokenizer.index_word[nUnkWordIndex] = strUnkToken

	# add <pad> <start> amnd <end> tokens for nmt later
	nPadWordIndex = len(tokenizer.word_index) + 1
	tokenizer.word_index['<pad>'] = nPadWordIndex
	tokenizer.index_word[nPadWordIndex] = '<pad>'

	nStartWordIndex = len(tokenizer.word_index) + 1
	tokenizer.word_index['<start>'] = nStartWordIndex
	tokenizer.index_word[nStartWordIndex] = '<start>'

	nEndWordIndex = len(tokenizer.word_index) + 1
	tokenizer.word_index['<end>'] = nEndWordIndex
	tokenizer.index_word[nEndWordIndex] = '<end>'

	return tokenizer

def apply_tokenization( list_sents, tokenizer, max_sent_length, reverse_seq, logger ) :
	nStartWordIndex = tokenizer.word_index['<start>']
	nEndWordIndex = tokenizer.word_index['<end>']
	nPadWordIndex = tokenizer.word_index['<pad>']
	
	# make a tensor and pad it to a target size
	tensor = tokenizer.texts_to_sequences( list_sents )
	for nIndexSent in range(len(tensor)) :

		# insert start token
		tensor[nIndexSent].insert( 0,nStartWordIndex )

		# make sure end token is not over max seq length so every training entry has an end token
		if len(tensor[nIndexSent]) < max_sent_length :
			tensor[nIndexSent].append( nEndWordIndex )
		else :
			tensor[nIndexSent][max_sent_length-1] = nEndWordIndex

		# reverse including start and end tokens, so we get "<end> ... blah ... <start>" fed as source
		if reverse_seq == True :
			tensor[nIndexSent].reverse()

	tensor = tf.keras.preprocessing.sequence.pad_sequences( sequences = tensor, maxlen = max_sent_length, padding='post', truncating='post', dtype = 'int32', value = nPadWordIndex )
	return tensor

def read_alignment_matrix( file, logger ) :

	if os.path.exists( file ) == False :
		raise Exception( 'alignment file does not exist - ' + repr(file) )

	dict_alignment = {}

	read_handle = codecs.open( file, 'r', 'utf-8', errors = 'strict' )
	str_content = read_handle.read()
	read_handle.close()
	list_lines = str_content.split( '\n' )
	if len(list_lines) < 2 :
		raise Exception( 'alignment matrix parse fail (< 2 lines) - ' + repr(file) )

	# (0 = source, 1 = target)
	nSentIndex = 0
	for line in list_lines :
		# newline at end
		if len(line) == 0 :
			continue
		dict_align = {}

		list_alignments = line.strip().split(' ')
		if len(list_alignments) > 0 :
			for str_align in list_alignments :
				list_tokens = str_align.split('-')
				if len(list_tokens) != 2 :
					raise Exception( 'alignment matrix parse fail (align pair failed to parse) - ' + repr(str_align) + ' : ' + repr(line) )
				
				# source token -> ( target token, ... )
				# there can be several target tokens aligned to a single source token
				nToken0 = int( list_tokens[0] )
				nToken1 = int( list_tokens[1] )
				if not nToken0 in dict_align :
					dict_align[ nToken0 ] = []
				dict_align[ nToken0 ].append( nToken1 )

		dict_alignment[nSentIndex] = dict_align
		nSentIndex += 1

	return dict_alignment

def create_lookup_dict( align_matrix, bitext, freq_threshold, logger ) :
	dict_lookup = {}
	if len(align_matrix) != len(bitext) :
		raise Exception( '# sents in alignment matrix (' + str(len(align_matrix)) + ') != # sents in bitext (' + str(len(bitext)) + ')' )

	# (0 = source, 1 = target) e.g. ru,en
	nSentIndex = 0
	for list_sent_pair in bitext :
		list_tokens0 = list_sent_pair[0].split(' ')
		list_tokens1 = list_sent_pair[1].split(' ')

		dict_align = align_matrix[nSentIndex]
		for nToken0 in dict_align :
			str_token0 = list_tokens0[nToken0]

			listToken1 = dict_align[nToken0]
			for nToken1 in listToken1 :
				str_token1 = list_tokens1[nToken1]

				if not str_token0 in dict_lookup :
					dict_lookup[str_token0] = {}
				if not str_token1 in dict_lookup[str_token0] :
					dict_lookup[str_token0][str_token1] = 1
				else :
					dict_lookup[str_token0][str_token1] += 1

		nSentIndex += 1

	# filter dict so (a) we only accept alignments with freq > 100 and (b) we take the highest freq alignment if there are several options (c) source and target are valid words (e.g. not numbers)
	# also only allow real words to appear in translation dict, not digits, symbols,  alphanumerics etc
	# replace { token0_optionA : freq, token0_optionB : freq ... } with the best token0_option
	list_del_token0 = []
	for str_token0 in dict_lookup :
		n_best_freq = None
		str_best_token1 = None

		if str_token0.isalpha() == True :
			for str_token1 in dict_lookup[str_token0] :

				if str_token1.isalpha() == True :
					nFreq = dict_lookup[str_token0][str_token1]
					if (n_best_freq == None) or (nFreq > n_best_freq) :
						n_best_freq = nFreq
						str_best_token1 = str_token1

		if (n_best_freq != None) and (n_best_freq > freq_threshold) :
			dict_lookup[str_token0] = str_best_token1
		else :
			list_del_token0.append( str_token0 )

	# remove any entry with no viable translation (e.g. < freq threshold or non-alpha)
	for str_token0 in list_del_token0 :
		del dict_lookup[str_token0]

	# all done
	return dict_lookup

def unkpos_replacement( align_matrix, list_source_sents, list_target_sents, source_tokenizer, target_tokenizer, logger ) :

	# use alignment_matrix to perform unk/unkpos replacement on corpus
	# if source = use <unk>
	# if target = lookup aligned source token, compute offset N, use <unkposN>. use <unkpos0> default for words with no unalignment
	for nSentIndex in range(len(list_source_sents)) :

		# replace source OOV tokens with '<unk>'
		list_token0 = list_source_sents[nSentIndex].split(' ')
		for nToken0 in range(len(list_token0)) :
			str_token = list_token0[nToken0]
			if not str_token in source_tokenizer.word_index :
				list_token0[nToken0] = '<unk>'
		list_source_sents[nSentIndex] = ' '.join( list_token0 )

		# replace target OOV tokens with '<unkposN>'
		if list_target_sents != None :
			dict_align = align_matrix[nSentIndex]
			list_token1 = list_target_sents[nSentIndex].split(' ')
			for nToken1 in range(len(list_token1)) :
				str_token = list_token1[nToken1]
				if not str_token in target_tokenizer.word_index :
					# check to see if this target token has an alignment to a source token
					# if so compute source token pos relative to the target token
					nPos = None
					for nToken0 in dict_align :
						if nToken1 in dict_align[nToken0] :
							nPos = nToken0 - nToken1
							break

					# unkposN replace
					if nPos == None :
						list_token1[nToken1] = '<unkpos0>'
					else :
						# any alignment outside range is replaced with unkpos0
						if nPos > 7 :
							nPos = 0
						if nPos < -7 :
							nPos = 0
						list_token1[nToken1] = '<unkpos' + str(nPos) + '>'
			list_target_sents[nSentIndex] = ' '.join( list_token1 )

def lookup_unkposN( sent_list_source = [], sent_list_translated = [], dict_lookup = {} ) :

	if len(sent_list_source) != len(sent_list_translated) :
		raise Exception( 'source (' + str(len(sent_list_source)) + ') and translated (' + str(len(sent_list_translated)) + ') sent lists have different sizes' )

	for nSentIndex in range(len(sent_list_translated)) :
		bChanged = False
		list_tokens0 = sent_list_source[nSentIndex].split(' ')
		list_tokens1 = sent_list_translated[nSentIndex].split(' ')

		for nToken1 in range(len(list_tokens1)) :
			if list_tokens1[nToken1].startswith('<unkpos') :
				str_pos = list_tokens1[nToken1][ len('<unkpos') : -1 ]
				nPos = int( str_pos )
				if (nPos < -7) or (nPos > 7) :
					nPos = 0

				nToken0 = nToken1 + nPos

				# if predicted unkposN points outside allowed source sent range
				# then default to pos = 0
				if (nToken0 < 0) or (nToken0 >= len(list_tokens0)) :
					if (nToken1 >= len(list_tokens0)) :
						nToken0 = None
					else :
						nToken0 = nToken1

				if nToken0 != None :
					# get translation of aligned source token to replace unkposN (if we can)
					str_token0 = list_tokens0[nToken0]
					if str_token0.isalpha() == True :
						# lookup translation for words. if we have a translation use it, if not use source word (identity mapping)
						if str_token0 in dict_lookup :
							str_translated_token0 = dict_lookup[str_token0]
						else :
							str_translated_token0 = str_token0
					else :
						# identity mapping for non-words (digits, symbols and alphanumerics)
						str_translated_token0 = str_token0

					list_tokens1[nToken1] = str_translated_token0
					bChanged = True
				else :
					# if target unkpos points outside source range AND target token itself outside source range then give up and use <unk> token
					list_tokens1[nToken1] = '<unk>'
					bChanged = True

		if bChanged == True :
			sent_list_translated[nSentIndex] = ' '.join( list_tokens1 )
