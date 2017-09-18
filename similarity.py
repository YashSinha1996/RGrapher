import pickle
with open("learnt-tf.pck","r+b") as ltf:
    learnt=pickle.load(ltf)

print("Beginning pickeling")
with open("wght-dict.pck","r+b") as ltf:
    word_doc_index=pickle.load(ltf)

print("Beginning pickeling")
with open("tfid.pck","r+b") as ltf:
    transd=pickle.load(ltf)	

def orderd_imp(word):
    index=word_doc_index[word]
    return [tup[1] for tup in sorted([(transd[index,i],i) for i in transd[index].nonzero()[1]])]

def similar(defin1,defin2):
	return transd[word_doc_index[defin1]].dot(transd[word_doc_index[defin2]].transpose())[0,0]

import itertools


if __name__ == '__main__':
    import itertools as it
    import json
    import numpy as np
    with open("dictionary.json","r") as ted_def:
        terms_desc=json.load(ted_def)
    key_list=terms_desc.keys()
    print(len(key_list))
    graph=np.zeros((len(key_list),len(key_list)),dtype=float)

    for pair in it.combinations(range(len(key_list)),2):
        graph[pair[0],pair[1]]=similar(key_list[pair[0]],key_list[pair[1]])
        graph[pair[1],pair[0]]=graph[pair[0],pair[1]]
    np.save("similar",graph)
    print(similar("PINTO","DEFIGURE"))
