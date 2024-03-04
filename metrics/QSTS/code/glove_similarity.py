#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 18:52:17 2021

@author: sdas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:34:56 2019

@author: sdas
"""

import numpy as np
import scipy.spatial
import string
import pickle
from tqdm import tqdm



embedding_file1='/home/sdas/embeddings/glove.840B.300d.txt'
embedding_file2='/home/sdas/embeddings/squadq-vectors.txt'
output_file='/home/sdas/embeddings/glove.840B.300d.wiki_ss.pkl'
terms_file='/home/sdas/wikipedia/wikiterms_stopwords.txt'
def make_embedding(embedding_file, output_file, terms_file):
    
    word2idx={}
    word2idx['UNK']=0
    lines = open(terms_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word = line.strip()
        word2idx[word]=len(word2idx)
    
    found = 0
    embedding = np.zeros((len(word2idx), 300), dtype=np.float32)    
    lines = open(embedding_file, "r", encoding="utf-8").readlines()
    for line in tqdm(lines):
        word_vec = line.split(" ")
        word = word_vec[0]
        if word in word2idx:
            idx = word2idx[word]
            vec = np.array(word_vec[1:], dtype=np.float32)
            embedding[idx] = vec
            found += 1
    
    embedding[0].fill(1/300)
    
    with open(output_file, "wb") as f:
        pickle.dump([embedding, word2idx], f)
        f.close()
        
    return 

def load_embeddings(inpfile):
    with open(inpfile, "rb") as f:
        [embedding, word2idx] = pickle.load(f)
        f.close()
        
    return embedding, word2idx

def getDistances(E, S):
 
    h = scipy.spatial.distance.cosine(E,S)
    return h

def checkpresence(sentence, model):
    
    newsent=""
    words = sentence.split()
    for word in words:
        
        word = word.lower().translate(str.maketrans('', '', string.punctuation))
        if word in model.vocab:
            newsent +=" "+word

    return newsent.strip()


if __name__=="__main__":
    
    wikiterms_glvembeddingsf='/home/sdas/embeddings/glove.840B.300d.wiki_ss.pkl'
    embeddings, word2idx = load_embeddings(wikiterms_glvembeddingsf)
    print(len(embeddings))
    print(len(word2idx))
    
    wordpairs=[
            ["lincoln", "columbus"],
            ["provide", "provided"], ["who", "when"],
           ["long", "date"], ["who", "author"],
           ["location", "country"], ['name', 'known']
          ]
    
    for wp in wordpairs:
        
        [word1, word2] = wp
        
        w1vector=embeddings[0]
        if word1 in word2idx:
            w1vector = embeddings[word2idx[word1]]
        
        w2vector=embeddings[0]
        if word2 in word2idx:
            w2vector = embeddings[word2idx[word2]]
        
            
        print ("("+word1+", "+word2+") = "+str(1-scipy.spatial.distance.cosine(w1vector, w2vector)))