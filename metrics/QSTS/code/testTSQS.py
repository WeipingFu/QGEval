#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:28:07 2022

@author: sdas
"""

import TypeSensitiveQSimilarity as qsim



q1s=[
     "What was the title of Bob Dylan's first album?",
     "What was the name of Vincent's brother?",
     "Where in Germany was the composer Beethoven born?",
 "Who was Columbus?",
"What was the name of Vincent's brother?",
"When did Freddie Mercury die?",
"How does one become an angel investor?"]


q2s=[ 
     
     "What was Bob Dylan's first album called?",
     "Who was Vincent's brother?",
     "Which city in Germany is the place of birth of Beethoven?",
     "Where is Columbus?",
     "What was the name of Vincent's painting?",
     "How did Freddie Mercury die?",

     "How do I get a job at Goldman Sachs?"


     ]

ph=[
   
    "After the eponymous first album, Bob Dylan went on to become the breakthrough songwriter of 'The Freewheelin'.",
   "Vincent's brother, Theo disagreed vehemently with "\
"the placement of Irises.",\
"The composer ludwig van beethoven went deaf in his final years.",\
"Lincoln the 16th president of the United State was born in Kentucky.",
"Vincent's brother, Theo disagreed vehemently "\
"with the placement of Irises.",
"On the evening of 24 November 1991, about 24 hours "\
"after issuing the statement, Mercury died at the age of "\
"45 at his home in Kensington",
""

]

q1types=["ENTY:cremat", "HUM:ind", "LOC:other", \
         "HUM:desc","HUM:ind","NUM:date", "ENTY:other"]

q2types=["ENTY:cremat", "HUM:ind", "LOC:city", \
         "LOC:other", "ENTY:cremat", "DESC:desc", "ENTY:other"]

for qx, q1 in enumerate(q1s):

    q2 = q2s[qx]
    passage = ph[qx]
    print ()
    print ("Q1: "+q1)
    print ("Q2: "+q2)
    print ("PASSAGE: " +passage)
    print ("QQ")
    print (qsim.scoreQPair(q1, q2, False, q1types[qx], q2types[qx]))
    
    print ("Q1P")
    print (qsim.scoreQPair(q1, passage, True))
    print ("Q2P")
    print (qsim.scoreQPair(q2, passage, True))