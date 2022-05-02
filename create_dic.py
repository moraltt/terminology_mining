'''
Go through all rows in tsv Wikimatrix file with Score > 1.04, perform quality tests for sentences.
The good sentences get lemmatized and their lemmas get stored in dictionary dic with a counter of occurences
The lemmatized sentences are stored in list sentence_lemmas. This avoids double lemmatization in posterior steps and saves much time, since this is the most time-consuming step.

Returns: 
· dic : dictionary storing all lemmatized tokens and the number of times they appear in the corpus.
· sentence_lemmas : list containing lemmatized English sentence in row i if sentence i was good, and None if sentence was skipped. 
'''

import gzip
import csv
import time
from utils import *
import globals as g


def Run(WikiMatrix):
    sentence_lemmas = []
    dic = {}
    start_time = time.time()
    threshold = 1e36

    lang1, lang2 = g.lang1, g.lang2

    with gzip.open(WikiMatrix, 'rt') as f:

        tsv_reader = csv.reader(f, delimiter="\t")
        i,j = -1,0
        
        for row in tsv_reader:
            if float(row[0]) < 1.04: break

            #Print progress
            if i%100000 == 0 and i!=0:
                print("Encoding of "+ str(j/1000000)+ " mill out of "+ str(i/1000000) + " mill sentences done after {:.2f} sec. ".format(time.time() - start_time),"Score: {:.2f}".format(float(row[0])))
            
            i+=1
            #Quicktests for skipping unvaluable rows
            if len(row)<3: continue #Skip untranslated
            if len(row[1])>400: continue #Skip too long
            if AnyMultiples(row[1]): continue #Skip wrongly parsed
            
            
            row[1:] = [preprocess(text) for text in row[1:]] #Preprocess sentences

            #Indentify english and translation
            if lang1 == 'en': t1 = row[1]; t2 = row[2]
            else: t1 = row[2]; t2 = row[1]

            #Append to dics
            if  LanguageIsCorrect(t1, t2, lang1, lang2):
                dic, row_lemma = UpdateDict(dic, t1, t2)
                sentence_lemmas.append(row_lemma)
                j+=1

            else: sentence_lemmas.append(None)  
        
                
    print("Encoding of "+ str(j/1000000)+ " mill out of "+ str(i/1000000) + " mill sentences done after {:.2f} sec".format(time.time() - start_time))    
    # Sort by frequency
    dic = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))       
    
    return dic, sentence_lemmas
