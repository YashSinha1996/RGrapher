import json
with open("dictionary.json","r") as ted_def:
    terms_desc=json.load(ted_def)

from tempfile import TemporaryFile as tf
files=[]
word_doc_index={}
for i,(term_key,term_des) in enumerate(terms_desc.items()):
    if len(term_key)>1:
        # filer=tf("r+")
        # filer.write(terms_desc[term_key])
        # filer.seek(0)
        files.append(term_des)
        word_doc_index[term_key]=i


# for filer in files:
#     print(filer.read())
#     filer.seek(0)

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf

learnt=tfidf("content")

transd=learnt.fit_transform(files)

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


def tf_tdf_sent(sent,learnt):
    tfs=tf(sent)
    #print(tfs)
    tfs_sorted=[]
    for word in tfs.keys():
        try:
            tfs[word]=tfs[word]*learnt.idf_[learnt.vocabulary_[word]]
        except Exception:
            tfs[word]=0
        tfs_sorted.append((tfs[word],word.lower()))
    tfs_sorted=sorted(tfs_sorted)
    tfs_sorted.reverse()
    return (tfs,tfs_sorted)

print(transd)
print(type(transd))

wighted_dict={}
key_list=terms_desc.keys()
no_keys=len(key_list)
print(no_keys)
# printProgressBar(0, no_keys, prefix = 'Progress :', suffix = 'Complete', length = 80)
# for i,defin in enumerate(key_list):
#     printProgressBar(i, no_keys, prefix = 'Progress :', suffix = 'Complete', length = 80)
#     wighted_dict[defin]=tf_tdf_sent(terms_desc[defin],learnt)[1]
    # print(defin,wighted_dict[defin])

idf={term:learnt.idf_[index] for term,index in learnt.vocabulary_.items()}
with open("terms_tfidf.json","w") as word_idf:
    json.dump(idf,word_idf)

from nltk.book import *
freq_std=FreqDist(text1)


# print(wighted_dict)
print("Beginning pickeling of learnt")
import pickle
with open("learnt-tf.pck","w+b") as ltf:
    pickle.dump(learnt,ltf)

print("Beginning pickeling")
with open("wght-dict.pck","w+b") as ltf:
    pickle.dump(word_doc_index,ltf)

print("Beginning pickeling")
with open("tfid.pck","w+b") as ltf:
    pickle.dump(transd,ltf)
