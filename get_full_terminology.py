'''
Deliver file storing:

source_term     target_term     score       source_sentence        target_sentence"

for 1-grams, 2-grams and 3-grams.
'''


import torch
import transformers
import itertools
import re
import gzip
from translate.storage.tmx import tmxfile
from transformers import AutoModel, AutoTokenizer
import argparse
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cossim 
from ftfy import fix_encoding
import read_tmx
from sentence_transformers import SentenceTransformer, models

parser = argparse.ArgumentParser()
parser.add_argument('fn', help='Name of the source file')
args = parser.parse_args()

filename = args.fn

# Create metadata df
f = gzip.open(filename+'.tmx.gz', 'rt', encoding='utf-8-sig')
metadata, df = read_tmx.read_tmx(f)
print('TMX processed')

# Model for alignments
model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased", device='cuda')
embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cuda')

print('Transformers loaded. Start extracting alignments')

# Extract terminology pairs 
i = 0
start_time = time.time()
Data = []
list_of_pairs = []
total_pairs = len(df)

for node in df:
    src, tgt = preprocess(df['source_sentence']),  preprocess(df['target_sentence']) # Preprocess sentences
    alignment = get_align(src, tgt) # Get alignment
    
    phrases = list(phrase_extraction(src, tgt, alignment)) # Extract corresponding phrases

    pairs, list_of_pairs = extract_pairs(phrases, list_of_pairs) # Select best quality pairs
    for pair in pairs: 
      Data.append([pair +[standarize(df['source_sentence']), standarize(df['target_sentence'])]])
    
    t = time.time()-start_time
    i+=1

    if i%1000 < 0.1: print(i, ' pairs out of ', total_pairs,  ' processed in ', round(t/60), 'min')

# Create Terminology df
df_Terminology = pd.DataFrame(Data, columns = ["source_term", "target_term", "score","source_sentence","target_sentence"])
    
    
# Merge
df_Full=df_Terminology.merge(df, on=['source_sentence'], how='inner')
#Save
df_Full.to_csv('Terminologie_Full-{}.txt'.format(filename), sep='\t',index=False)
print('{} finished in {} seconds'.format(filename, round(time.time()-timestart)))
