#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:18:57 2021

@author: sdas
"""
import sys
import numpy as np
import scipy.spatial
import StanzaForQuestions as stanzaproc
import glove_similarity 

from nltk.corpus import stopwords



class QSTS:
    def __init__(self, wikiterms_glvembeddingsf):
        self.stop_words = set(stopwords.words('english'))
        self.embeddings, self.word2idx = glove_similarity.load_embeddings(wikiterms_glvembeddingsf)

    def getGlSim(self, word1, word2):
        if word1==word2:
            return 1
        elif word1 in self.word2idx and word2 in self.word2idx:
        
            w1vector = self.embeddings[self.word2idx[word1]]
            w2vector = self.embeddings[self.word2idx[word2]]
            return (1-scipy.spatial.distance.cosine(w1vector, w2vector))
        else:
            return 0

    def separateTokens(self, question):
        qwords, qpos, qner, qdbigs = stanzaproc.getNLPInfo2(question)
        qnames=[]
        for px, pos in enumerate(qpos):
            if pos=="PROPN" or qner[px]!="O":
                (_, w, _, _, _) = qdbigs[px]
                qnames.append(w)
    
        return qnames, qdbigs

    def _scoreQPair(self, q1, q2):

        gnames, gdb = self.separateTokens(q1)
        pnames, pdb = self.separateTokens(q2)
        
        namematches=[]
        for gname in gnames:
        
            found=False
            
            for pname in pnames:    
                if pname==gname:
                    found=True
                    break
            
            if found:
                namematches.append(1)
            else:
                namematches.append(0)
        
    #    print (gnames)
    #    print (pnames)
        if len(namematches)==0:
            score1 = 1
        else:
            score1 = np.count_nonzero(namematches)/len(namematches)
        
        countvalid=0
        gsims=[]
        gsiminds=[]
        for gx, gele in enumerate(gdb):
            (g1pos, g1, grel, g2pos, g2) = gele
            maxsim=0
            maxindx=-1
            
            if grel in stanzaproc.discarddep \
            or g1 in self.stop_words or g2 in self.stop_words:
                continue
            
            countvalid += 1
    #        print ("Valid "+str(gele))
            for px, pele in enumerate(pdb):
                (p1pos, p1, prel, p2pos, p2) = pele
                score = 0
                if grel==prel:
                    if g1pos=="PROPN" and g2pos!="PROPN":
                        
                        if p1==g1:
                            score = 1
                        
                        score *= self.getGlSim(g2, p2)
                    elif g1pos!="PROPN" and g2pos=="PROPN":
                        
                        if p2==g2:
                            score = 1
                        score *= self.getGlSim(g1, p1)
                    
                    elif g1pos=="PROPN" and g2pos=="PROPN":
                        if p2==g2 and p1==g1:
                            score = 1
                    else:
                        score = 0.5*(self.getGlSim(g1, p1) +  self.getGlSim(g2, p2))
                    
                    if score>maxsim:
                        maxsim=score
                        maxindx=px
            
            
            if maxindx!=-1:
                gsims.append(maxsim)
                gsiminds.append((gx, maxindx))
    #            print ("Best match "+str(gx)+" "+str(maxindx))
    #            print (gdb[gx])
    #            print (pdb[maxindx])
    #
    #    print (gsims)
    #    print (countvalid)
        if countvalid==0:
            return score1, 1
        if len(gsiminds)==0 and countvalid>0:
            return score1, 0
        return score1, np.sum(gsims)/countvalid
     

    def getQCScore(self, q1qc, q2qc):
        
        q1cc=q1qc.split(":")[0]
        q1fc=q1qc.split(":")[1]
        q2cc=q2qc.split(":")[0]
        q2fc=q2qc.split(":")[1]
        
        if q1qc==q2qc:
            qcmatch = 1
        elif q1cc==q2cc and (q1fc=="other" or q2fc=="other"):
            qcmatch = 0.75
        elif q1cc==q2cc:
            qcmatch = 0.5
        else:
            qcmatch = 0
            
        return qcmatch

    def scoreQPair(self, goldq, predq, ignoreQC=True, goldqc=None, predqc=None):

        ns, os = self._scoreQPair(goldq, predq)
        
        if ignoreQC:
            qcscore = 1
        else:
            qcscore = self.getQCScore(goldqc, predqc)
            
        return np.power(qcscore*ns*os, 1/3)
            
    def computeQSimilarity(self, ref_file, hyp_file, outfile, useQC):   
        fout = open (outfile, "w")
        rlines = open (ref_file, "r").readlines()
        hlines = open (hyp_file, "r").readlines()
        scores = []
        for rx, rline in enumerate(rlines):
            
            hline = hlines[rx].strip()
            
            if useQC:
                refqc = rline.strip().split("\t")[0]
                ref = rline.strip().split("\t")[1]
                hypqc = hline.strip().split("\t")[0]
                hyp = hline.strip().split("\t")[1]
                score = self.scoreQPair(ref, hyp, False, refqc, hypqc)
            else:
                ref = rline.strip()
                hyp = hline.strip()
                score = self.scoreQPair(ref, hyp)
                
            fout.write(ref.strip()+"\t"+hyp+"\t"+str(score)+"\n")
            fout.flush()
            scores.append(score)
            
        fout.write("Aggregate_Scores\t"+str(np.mean(scores))+"\t"+str(np.std(scores))+"\n")
        fout.close()
        print ("Total Question Pairs "+str(len(rlines)))
        print("Aggregate_Scores\t"+str(np.mean(scores))+"\t"+str(np.std(scores))+"\n")
        return



if __name__=="__main__":
    
    print ("\nNOTE: Inpfiles should tab-separated with col-1 having Qtype information for the type-sensitive version\n")
    if len(sys.argv)<4:
        print ("args1: ref/questions-file \n"
               "args2: hyp/passages-file \n"
               "args3: out-file \n"
               "[args4: isQTypeSensitive=True/False, default=False]")
    
    else:
        
        rfile = sys.argv[1]
        hfile = sys.argv[2]
        ofile = sys.argv[3]
        
        if len(sys.argv)>4 and sys.argv[4].lower().strip=="true":
            qt = True
        else:
            qt = False
        
        wikiterms_glvembeddingsf='../../embeddings/glove.840B.300d.wiki_ss.pkl'
        qsts = QSTS(wikiterms_glvembeddingsf)
        qsts.computeQSimilarity(rfile, hfile, ofile, qt)