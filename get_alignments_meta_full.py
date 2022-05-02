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



def phrase_extraction(srctext, trgtext, alignment):

    def extract(f_start, f_end, e_start, e_end):
        if f_end < 0:  # 0-based indexing.
            return {}
        # Check if alignement points are consistent.
        for e,f in alignment:
            if ((f_start <= f <= f_end) and
               (e < e_start or e > e_end)):
                return {}

        # Add phrase pairs (incl. additional unaligned f)
        # Remark:  how to interpret "additional unaligned f"?
        phrases = set()
        fs = f_start
        # repeat-
        while True:
            fe = f_end
            # repeat-
            while True:
                # add phrase pair ([e_start, e_end], [fs, fe]) to set E
                # Need to +1 in range  to include the end-point.
                src_phrase = " ".join(srctext[i] for i in range(e_start,e_end+1))
                trg_phrase = " ".join(trgtext[i] for i in range(fs,fe+1))
                # Include more data for later ordering.
                phrases.add(((e_start, e_end+1), (f_start, f_end+1), src_phrase, trg_phrase))
                fe += 1 # fe++
                # -until fe aligned or out-of-bounds
                if fe in f_aligned or fe == trglen:
                    break
            fs -=1  # fe--
            # -until fs aligned or out-of- bounds
            if fs in f_aligned or fs < 0:
                break
        return phrases

    # Calculate no. of tokens in source and target texts.
    srctext = srctext.split()   # e
    trgtext = trgtext.split()   # f
    srclen = len(srctext)       # len(e)
    trglen = len(trgtext)       # len(f)
    # Keeps an index of which source/target words are aligned.
    e_aligned = [i for i,_ in alignment]
    f_aligned = [j for _,j in alignment]

    bp = set() # set of phrase pairs BP
    # for e start = 1 ... length(e) do
    # Index e_start from 0 to len(e) - 1
    for e_start in range(srclen):
        # for e end = e start ... length(e) do
        # Index e_end from e_start to len(e) - 1
        for e_end in range(e_start, srclen):
            # // find the minimally matching foreign phrase
            # (f start , f end ) = ( length(f), 0 )
            # f_start ∈ [0, len(f) - 1]; f_end ∈ [0, len(f) - 1]
            f_start, f_end = trglen-1 , -1  #  0-based indexing
            # for all (e,f) ∈ A do
            for e,f in alignment:
                # if e start ≤ e ≤ e end then
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            # add extract (f start , f end , e start , e end ) to set BP
            phrases = extract(f_start, f_end, e_start, e_end)

            ngram = e_end - e_start < 2 # Only 1-gram or 2-gram
            if phrases and ngram:
                bp.update(phrases)
    return bp



def preprocess(string):
  start = 0
  MATCH = re.search('[a-zA-ZÀ-ÖØ-öø-ÿ0-9]', string)
  if MATCH is not None: start = MATCH.span()[0]
  document = string[start:].strip()
  document = re.sub('[^a-zA-ZÀ-ÖØ-öø-ÿ]'," ", document)
  document = re.sub('  +'," ", document) #Multiple spacing 
  document = document.strip()
  try: end = re.search("(?s:.*)[A-Za-z]", document).span()[1]; document = document[:end];return document 
  except:  print(document); return document 
  

def get_align(src, tgt):
  # pre-processing
  sent_src, sent_tgt = src.strip().split(), tgt.strip().split()

  token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in sent_tgt]
  wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
  ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt', model_max_length=tokenizer.model_max_length, truncation=True)['input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True, model_max_length=tokenizer.model_max_length)['input_ids']
  sub2word_map_src = []
  for i, word_list in enumerate(token_src):
    sub2word_map_src += [i for x in word_list]
  sub2word_map_tgt = []
  for i, word_list in enumerate(token_tgt):
    sub2word_map_tgt += [i for x in word_list]

  # alignment
  align_layer = 9
  threshold = 1e-4
  model.eval()
  with torch.no_grad():
    out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
    out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

    dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

    softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
    softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

    softmax_inter = (softmax_srctgt > threshold)*(softmax_tgtsrc > threshold)

  align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
  align_words = set()
  for i, j in align_subwords:
    align_words.add( (sub2word_map_src[i], sub2word_map_tgt[j]) )
  
  return sorted(align_words)

from sentence_transformers import SentenceTransformer, models


def extract_pairs(phrases, list_of_pairs):
  # Returns array storing [ [En, De, score], [En, De, score]  ]
  EN = [x[2] for x in phrases] # original EN ngrams
  DE = [x[3] for x in phrases] # translated DE ngrams

  pairs = []
  for x in set(EN):
    fil = [x == y for y in EN]
    candidates = list(itertools.compress(DE, fil)) # Candidate translations
    embeddings = embedder.encode([x]+candidates) # Embeddings
    scores = cossim(embeddings[0].reshape(1,-1), embeddings[1:])[0] #Compute cossim between source and alignments 

    pair = [x, candidates[np.argmax(scores)]]
    if max(scores)>.6 and pair not in list_of_pairs:
      list_of_pairs.append(pair)
      pairs.append( pair+[str(max(scores)) ]) # Extract best
      
  return pairs, list_of_pairs


parser = argparse.ArgumentParser()
parser.add_argument('fn', help='Name of the source file')
args = parser.parse_args()

filename = args.fn

# Create metadata df
RE_XML_ILLEGAL = u'([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])' + \
                 u'|' + \
                 u'([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])' % \
                  (chr(0xd800),chr(0xdbff),chr(0xdc00),chr(0xdfff),
                   chr(0xd800),chr(0xdbff),chr(0xdc00),chr(0xdfff),
                   chr(0xd800),chr(0xdbff),chr(0xdc00),chr(0xdfff))


f = gzip.open(filename+'.tmx.gz', 'rt', encoding='utf-8')
change = f.read().replace(RE_XML_ILLEGAL, "?") # Replace illegal xml characters before parsing
metadata, df = read_tmx.read_tmx(change)
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
      Data.append([pair +[df['source_sentence'], df['target_sentence']]])
    
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
