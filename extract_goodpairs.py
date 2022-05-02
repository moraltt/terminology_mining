'''
Go through rows in Wikimatrix dataset that were unskipped before. Check if lemmas present in English sentence are in reduced dictionary of unusual words
and, if so, extract row along with lemma in question as additional column.

Returns: 
· goodpairs: List of list with format: Score | Sentence Lang1 | Sentence Lang2 | Lemma 1 | Lemma 2 ... 
'''

import gzip
import csv
from utils import *
import globals as g

def Run(WikiMatrix, dic_red, lemmas):
    dic_check = {''}
    goodpairs = []
    start_time = time.time()
    allowprint = True

    lang1, lang2 = g.lang1, g.lang2
    
    with gzip.open(WikiMatrix, 'rt') as f:

        tsv_reader = csv.reader(f, delimiter="\t")
        i,j = -1,0
        
        for row in tsv_reader:
            if float(row[0]) < 1.04: break
            
            #Print progress
            if i%100000 == 0 and i!=0:
                print("Encoding of "+ str(j)+ " out of "+ str(i)+
                    " sentences done after {:.2f} sec. ".format(time.time() - start_time),"Score: {:.2f}".format(float(row[0])))
            
            i+=1


            #Quicktests for skipping unvaluable rows
            if type(lemmas[i]) is not str: continue #Skip non-matched
            if type(lemmas[i]) == "None": continue #Skip non-matched (in case lemmas is taken from txt)
            if len(row)<3: continue #Skip untranslated
            if len(row[1])>400: continue #Skip too long
            if AnyMultiples(row[1]): continue

            else: 
                #Indentify english and translation
                if lang1 == 'en': translation = row[2]
                else: translation = row[1]
                
                g.row = row
                ExtractSenteces(lemmas[i], translation); j+=1 

            if len(goodpairs) % 100 == 0 and len(goodpairs)>0 and allowprint: 
                print(len(goodpairs),'/', len(dic_red))
                allowprint = False
            if len(goodpairs) % 101 == 0: allowprint = True
            
                
            print("Encoding of "+ str(j/1000000)+ " mill out of "+ str(i/1000000)+
            " mill sentences done after {:.2f} sec".format(time.time() - start_time))   
            
            return goodpairs

