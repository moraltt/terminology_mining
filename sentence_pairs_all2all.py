'''
Need to install  fasttext and spacy. Download "en_core_web_sm".
Edit path and add path to WikiMatrix files 
Try it as GetSentences hr (I included Croatian as an example)
'''


#Imports
import os
import argparse
import json
import wget
import urllib

# os.chdir('../Wikimatrix') # You should change this

# Custom
from utils import *
import create_dic
import extract_goodpairs
import globals as g

if __name__ == "__main__": 
  g.initialize()

  #Possible languages
  LANGS = ['vi','ar','bs','bg','zh','hr','cs','da','nl','et','fi','fr','ka','de','el','he','hu','is','it','ja','kk','ko','lv','lt','no','pl','pt','ro','ru','sr','sk','sl','es','sv','th','tr','uk','zh']


  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('l1', help='LANG1: second language \nPossible languages are:\n{}'.format(' '.join(LANGS)))
  parser.add_argument('l2', help='LANG2: second language \nPossible languages are:\n{}'.format(' '.join(LANGS)))
  args = parser.parse_args()
 
  # Define languages in alphabetical order  ----------------------------------------------------------------------------------------------
  g.lang1, g.lang2 = sorted([str(args.l1), str(args.l2)])
  lang1, lang2 = g.lang1, g.lang2

  # Define languages by relevance  ------------------------------------------------------------------------------------------------------
  if g.Sizes[lang1] > g.Sizes[lang2]:  
      g.LANG1, g.LANG2 = lang1, lang2
  else: g.LANG1, g.LANG2 = lang2, lang1
  
  LANG1, LANG2 = g.LANG1, g.LANG2


  # Load WikiMatrix file  ---------------------------------------------------------------------------------------------------------------
  path = '' # Add path to WikiMatrix files
  WikiMatrix = path + 'WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2)

  # If file does not exist, download it 
  if not os.path.exists(WikiMatrix): 
    print('\nNo WikiMatrix file found. Downloading from net')
    wget.download('https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2))
    WikiMatrix = 'WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2)


  # Check if lemmas wordlist is avalible for main language  ------------------------------------------------------------------------------
  
  try: 
    wget.download('https://raw.githubusercontent.com/michmech/lemmatization-lists/master/lemmatization-{}.txt'.format(LANG1))
    
    print('\nWordlist is availible. Creating lemmatized version')
    with open('lemmatization-{}.txt'.format(LANG1)) as f:
      g.wordlist = []
      for line in f:
        if '\t' not in line: break
        index = line.index('\t')
        g.wordlist.append(line[:index])
    f.close()

  except urllib.error.HTTPError:  
    g.wordlist = [] 
    print('\nWordlist is not availible for this language. The corresponding step will be skipped')


  '''
  # Create frequency dictionary of lemmas  -------------------------------------------------------------------------------------------------
  if os.path.exists("dic_{}-{}.txt".format(lang1,lang2)):
    dic = load_from_txt(("dic_{}-{}.txt".format(lang1,lang2)))
    lemmas = load_from_txt(("lemmatized_sentences_{}-{}.txt".format(lang1,lang2)))
    print('\nDictionary and lemmas have been loaded')
  
  else: 
    print('\nNo dictionary found. Creating dictionary')
    dic, lemmas = create_dic.Run(WikiMatrix) 

    # Save dictionary and lemmas variable as .txt file 
    if dic != {}:
      print('\nSaving dictionary')
      SaveVar(data=dic, file=path+"dic_{}-{}.txt".format(lang1,lang2))
      SaveVar(data=lemmas, file=path+"lemmatized_sentences_{}-{}.txt".format(lang1,lang2))
    else: print('ERROR: DICTIONARY IS EMPTY!!!'); os._exit(0)
  '''
  print('\nNo dictionary found. Creating dictionary')
  dic, lemmas = create_dic.Run(WikiMatrix) 

  # Save dictionary and lemmas variable as .txt file 
  if dic != {}:
    print('\nSaving dictionary')
    SaveVar(data=dic, file=path+"dic_{}-{}.txt".format(lang1,lang2))
    SaveVar(data=lemmas, file=path+"lemmatized_sentences_{}-{}.txt".format(lang1,lang2))
  else: print('ERROR: DICTIONARY IS EMPTY!!!'); os._exit(0)


  # Reduce dictionary to lemmas with [m,n] occurrences  ------------------------------------------------------------------------------------
  # Default: n, m = 2, 9
  if len(dic.keys())>10000: g.dic_red = Create_dic_red(dic,2,9)
  else: g.dic_red = Create_dic_red(dic,1,9)

  # Extract sentences that fulfill all criteria   ------------------------------------------------------------------------------------------
  print('\nExtracting suitable pairs of sentences')
  goodpairs = extract_goodpairs.Run(WikiMatrix, g.dic_red, lemmas)

  # Save sentences tsv file as .txt  -------------------------------------------------------------------------------------------------------
  SaveVar(data=goodpairs, file=path+"Sentences_Freqs02-09_ScoreOver1.04_{}-{}.txt".format(lang1,lang2))
