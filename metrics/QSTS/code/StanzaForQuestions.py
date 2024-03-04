#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:30:08 2021

@author: sdas
"""


import stanza
nlp = stanza.Pipeline(lang='en', use_gpu=True, processors='tokenize,pos,lemma,ner,depparse')





openclassPOS={}
closedclassPOS={}
otherPOS={}

openclassPOS["ADJ"]=""
openclassPOS["ADV"]=""
openclassPOS["INTJ"]=""
openclassPOS["NOUN"]=""
openclassPOS["PROPN"]=""
openclassPOS["VERB"]=""

closedclassPOS["ADP"]=""
closedclassPOS["AUX"]=""
closedclassPOS["CCONJ"]=""
closedclassPOS["DET"]=""
closedclassPOS["NUM"]=""
closedclassPOS["PART"]=""
closedclassPOS["PRON"]=""
closedclassPOS["SCONJ"]=""

otherPOS["PUNCT"]=""
otherPOS["SYM"]=""
otherPOS["X"]=""

discarddep={}
discarddep["det"]=""
discarddep["expl"]=""
discarddep["goeswith"]=""
discarddep["possessive"]=""
discarddep["preconj"]=""
discarddep["predet"]=""
discarddep["prep"]=""
discarddep["punct"]=""
discarddep["ref"]=""

#########33

def tokenize(text):
    
    sentwords = []
    
   
    tagged = nlp (text)
    for sentence in tagged.sentences:
        
        for tx, token in enumerate(sentence.tokens):
           
            word = token.words[0]
            sentwords.append (word.text)
            
    
    return sentwords

def sentencify(text):
    
    sentwords = []
    postags = []
    nertags = []
  
   
    tagged = nlp (text)
    for sentence in tagged.sentences:
        w=[]
        p=[]
        n=[]
        
        for tx, token in enumerate(sentence.tokens):
           
            word = token.words[0]
            
            w.append (word.text)
            p.append (word.upos)
            n.append (token.ner)
        
        sentwords.append(w)
        postags.append(p)
        nertags.append(n)
            
    
    return sentwords, postags, nertags

def getNLPInfo(text):
    
    sentwords = []
    postags = []
    nertags = []
    dbigs=[]
   
    tagged = nlp (text)
    for sentence in tagged.sentences:
        for tx, token in enumerate(sentence.tokens):
           
            word = token.words[0]
            
            sentwords.append (word.text)
            postags.append (word.upos)
            nertags.append (token.ner)

            if word.head ==0:
                head="root"
                headx=-1
     #               headpos="ROOT"
            else:
                head = sentence.tokens[word.head-1].words[0].lemma
                headx = word.head-1
            
            dbigs.append((tx, word.lemma.lower(), word.deprel, headx, head.lower()))
            
    
    return sentwords, postags, nertags, dbigs


def getNLPInfo2(text):
    
    sentwords = []
    postags = []
    nertags = []
    dbigs=[]
   
    tagged = nlp (text)
    for sentence in tagged.sentences:
        for tx, token in enumerate(sentence.tokens):
           
            word = token.words[0]
            
            sentwords.append (word.text)
            postags.append (word.upos)
            nertags.append (token.ner)

            if word.head ==0:
                head="root"
                headpos="rootpos"
     #               headpos="ROOT"
            else:
                head = sentence.tokens[word.head-1].words[0].lemma
                headpos = sentence.tokens[word.head-1].words[0].upos
            
            dbigs.append((word.upos, word.lemma.lower(), word.deprel, headpos, head.lower()))
            
    
    return sentwords, postags, nertags, dbigs
#text="After the eponymous first album, Bob Dylan went on to become the \
#      breakthrough songwriter of 'The Freewheelin'."
#      
#s, p, n, db = getNLPInfo(text)
#print (s)
#print (p)
#print (n)
#print ()

