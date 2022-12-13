import torch
import transformers
import itertools
import re
import gzip
from translate.storage.tmx import tmxfile
from transformers import AutoModel, AutoTokenizer
import argparse
import time
from functions import * 

parser = argparse.ArgumentParser()
parser.add_argument('fn', help='Name of the source file')
args = parser.parse_args()

filename = args.fn

output_file = open('ALIGN-'+filename+'.txt', 'w') 
    
print('Output file created')


f = gzip.open(filename+'.tmx.gz', 'rt', encoding='utf-8')
T = tmxfile(f, 'en', 'de', encoding='utf-8')
print('TMX processed')

model = AutoModel.from_pretrained("dbmdz/bert-base-german-cased")
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
print('Transformers loaded. Start extracting alignments')

total_pairs = 0
for node in T.unit_iter():
  total_pairs +=1



i = 0
start_time = time.time()

for node in T.unit_iter():
    src, tgt = preprocess(node.source),  preprocess(node.target)
    alignment = get_align(src, tgt)
    phrases = phrase_extraction(src, tgt, alignment)
    output_file.write(' ||| '.join([src, tgt, str(alignment), str(phrases)])+'\n\n')

    t = time.time()-start_time

    i+=1
    if i%50000 < 0.1: print(i, ' pairs out of ', total_pairs,  ' processed in ', round(t/60), 'min')
    if t %(60*30) < 0.1: print(i, ' pairs out of ', total_pairs,  ' processed in ', round(t/60), 'min')
