import os
import sys
from spacy.en import English
from collections import defaultdict
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

def tf(sent):
    words=tfidf().build_tokenizer()(sent.lower())
    tf={}
    for word in words:
        if word in tf:
            tf[word]+=1
        else:
            tf[word]=1
    for key in tf.keys():
        tf[key]=tf[key]/len(words)
    return tf

def load_prominence(textfile):
	f = codecs.open(textfile, 'r', 'utf-8')
	out = defaultdict(int)
	for line in f:
		line = line.strip()
		cols = line.split('\t')
		term = cols[0]
		freq = cols[1]
		out[term] = freq
	f.close()
	return out

# def generate_dataset(*args):
# 	args = args[0]
# 	if len(args) == 2:
# 		text_file,label =args[0],args[1]
# 		print 'Processing:: ',text_file,' || With label:: ',label
# 		extract_features(text_file, label)
# 	elif len(args) > 2 and len(args) % 2 == 0:
# 		pairs = zip(args[::2], args[1::2])
# 		for text_file,label in pairs:
# 			print 'Processing:: ',text_file,' || With label:: ',label
# 			extract_features(text_file, label)
# 	else:
# 		sys.exit('Please provide an even number of $filename $label pairs as argument(s)')

def part_of_NP(toke,doc):
	for np in doc.noun_chunks:
		for index,word in np:
			if toke.i == word.i:
				if index > 0:
					return 1
				return 0
	return -1


def extract_features(defination):
	line = line.strip().lower()
	tfreq = tf(line)
	defin = nlp(line)
	sents = defin.sents
	out=[]
	for sent in sents:
		rich_sent = []
		for token in sent:
			rich_token = []
			rich_token.append(token.orth)
			rich_token.append(token.pos)
			rich_token.append(token.dep)
			rich_token.append(tfreq[token.text])
			rich_token.append(tfreq[token.text]*tf_idf[token.text])
			rich_token.append((freq_all[token.text]/len(all_words_def))-(freq_std[token.text]/len(text1)))
			if token.orth_ in definitional_prom:
				rich_token.append(int(definitional_prom[token.orth_]))
			else:
				rich_token.append(0)
			if token.orth_ in definiens_prom:
				rich_token.append(int(definiens_prom[token.orth_]))
			else:
				rich_token.append(0)

			rich_sent.append(rich_token)
			out.append(rich_token)
	return out

if __name__ == '__main__':

	# args = sys.argv[1:]
	#print 'Args: ',args
	import json
	dictionary = json.load(open("dictionary.json","r"))
	from nltk.book import *
	freq_std=FreqDist(text1)
	all_words_def = []
	for defs in dictionary.values():
		all_words_def.extend(tfidf().build_tokenizer()(defs.lower()))
	freq_all = FreqDist(all_words_def)
	print('Loading definitional prominence dict')
	definitional_prom = load_prominence('features_data/definitional_prominence_wordfreqs.txt')
	print('Loading definiens prominence dict')
	definiens_prom = load_prominence('features_data/definiens_prominence_wordfreqs.txt')
	print('Loading Spacy')
	tf_idf = json.load("tfidf.json")
	# termhood = json.load("termhood.json")
	nlp = English()
	extract_features(args)

