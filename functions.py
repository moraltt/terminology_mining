import torch
import transformers
import itertools
import re
import gzip
from translate.storage.tmx import tmxfile
from transformers import AutoModel, AutoTokenizer
import argparse
import time

def phrase_extraction(srctext, trgtext, alignment):
    "Phrase-extraction algorithm from NLTK"
    """
    Phrase extraction algorithm extracts all consistent phrase pairs from 
    a word-aligned sentence pair.
    The idea is to loop over all possible source language (e) phrases and find 
    the minimal foregin phrase (f) that matches each of them. Matching is done 
    by identifying all alignment points for the source phrase and finding the 
    shortest foreign phrase that includes all hte foreign counterparts for the 
    source words.
    In short, a phrase alignment has to 
    (a) contain all alignment points for all covered words
    (b) contain at least one alignment point
    A phrase pair (e, f ) is consistent with an alignment A if and only if:
    
    (i) No English words in the phrase pair are aligned to words outside it.
    
           ∀e i ∈ e, (e i , f j ) ∈ A ⇒ f j ∈ f
    
    (ii) No Foreign words in the phrase pair are aligned to words outside it. 
            
            ∀f j ∈ f , (e i , f j ) ∈ A ⇒ e i ∈ e
    
    (iii) The phrase pair contains at least one alignment point. 
            
            ∃e i ∈ e  ̄ , f j ∈ f  ̄ s.t. (e i , f j ) ∈ A
            
    [in]:
    *srctext* is the tokenized source sentence string.
    *trgtext* is the tokenized target sentence string.
    *alignment* is the word alignment outputs in pharaoh format
    
    [out]:
    *bp* is the phrases extracted from the algorithm, it's made up of a tuple 
    that stores:
        ( (src_from, src_to), (trg_from, trg_to), src_phrase, target_phrase )
    
    (i)   the position of the source phrase
    (ii)  the position of the target phrase
    (iii) the source phrase
    (iv)  the target phrase
    >>> srctext = "michael assumes that he will stay in the house"
    >>> trgtext = "michael geht davon aus , dass er im haus bleibt"
    >>> alignment = [(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9), 
    ... (5,9), (6,7), (7,7), (8,8)]
    >>> phrases = phrase_extraction(srctext, trgtext, alignment)
    >>> for i in sorted(phrases):
    ...    print i
    ...
    ((0, 1), (0, 1), 'michael', 'michael')
    ((0, 2), (0, 4), 'michael assumes', 'michael geht davon aus')
    ((0, 2), (0, 4), 'michael assumes', 'michael geht davon aus ,')
    ((0, 3), (0, 6), 'michael assumes that', 'michael geht davon aus , dass')
    ((0, 4), (0, 7), 'michael assumes that he', 'michael geht davon aus , dass er')
    ((0, 9), (0, 10), 'michael assumes that he will stay in the house', 'michael geht davon aus , dass er im haus bleibt')
    ((1, 2), (1, 4), 'assumes', 'geht davon aus')
    ((1, 2), (1, 4), 'assumes', 'geht davon aus ,')
    ((1, 3), (1, 6), 'assumes that', 'geht davon aus , dass')
    ((1, 4), (1, 7), 'assumes that he', 'geht davon aus , dass er')
    ((1, 9), (1, 10), 'assumes that he will stay in the house', 'geht davon aus , dass er im haus bleibt')
    ((2, 3), (5, 6), 'that', ', dass')
    ((2, 3), (5, 6), 'that', 'dass')
    ((2, 4), (5, 7), 'that he', ', dass er')
    ((2, 4), (5, 7), 'that he', 'dass er')
    ((2, 9), (5, 10), 'that he will stay in the house', ', dass er im haus bleibt')
    ((2, 9), (5, 10), 'that he will stay in the house', 'dass er im haus bleibt')
    ((3, 4), (6, 7), 'he', 'er')
    ((3, 9), (6, 10), 'he will stay in the house', 'er im haus bleibt')
    ((4, 6), (9, 10), 'will stay', 'bleibt')
    ((4, 9), (7, 10), 'will stay in the house', 'im haus bleibt')
    ((6, 8), (7, 8), 'in the', 'im')
    ((6, 9), (7, 9), 'in the house', 'im haus')
    ((8, 9), (8, 9), 'house', 'haus')
    """
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
            if phrases:
                bp.update(phrases)
    return bp


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


def preprocess(string):
  start = 0
  MATCH = re.search('[a-zA-ZÀ-ÖØ-öø-ÿ]', string)
  if MATCH is not None: start = MATCH.span()[0]
  return string[start:].strip()
