# Terminology mining through semantic search applied to AbbVie Translation Memories

## The goal: Model distillation

Ideally we would have infinite resources (eg. Google)

Low-resource alternative: Optimize the dataset for model refinement

**Ansatz:** Let the model “see“ examples at least once. 

**Our task:** Mine rare words. Find example sentences and and their translations.

## The WikiMatrix Dataset

The Wikimatrix* dataset aims to mine parallel sentences from Wikipedia articles in different language pairs.
WikiMatrix.LANG_1-LANG_2.tsv  contains sentences in LANG_1 and their counterparts in LANG_2

Sentence pairs are classified by score** measuring quality of correspondance.

\** Maximum margin criterion from **Haifeng Li, Tao Jiang and Keshu Zhang, "Efficient and robust feature extraction by maximum margin criterion,"  IEEE Transactions on Neural Networks (2006)**

## Extraction of suitable pairs

1) Identify LANG_1 as preferent based on resources availible.

2) Create dictionary frequency of lemmas* for LANG_1.

3) Reduce dictionary to instances occurring (n,m) times not too frequent, not extremely rare (possible mispellings, wrong lemmatizations, etc.)

4) Revisit Wikimatrix and extract 1 sentence for each lemma 
```diff
- Eg: Furthermore, individual V1 neurons in humans and animals with binocular vision have ocular dominance 
Just one sencence  deals as example for three rare words
```
![image](https://user-images.githubusercontent.com/99658381/166222263-cc774077-edf1-4c32-a1f6-1bb074094870.png)

### Multilingual all-to-all implementation:
An all-to-all extension was built, altering the pipeline accounting for language-specific characteristics.
- Language-specific RegEX
- Different lemmatizers (Spacy, simplemma)
- Lemmatization may not apply (Chinese)
- Special tokenizers (konlpy, qalsadi, jieba, sudachipy)

## A pipeline for Word-alignment extraction
1) Preprocess sentences
2) Get possible alignments (1- or 2-gram to arbitrary)
3) Encode source n-gram and translations (Multilingual encoder)
4) Perform cosine similarity and keep best if score over 0.7

![image](https://user-images.githubusercontent.com/99658381/166222850-20be8c7d-fa8d-4e0c-b2e5-53070942aebd.png)
