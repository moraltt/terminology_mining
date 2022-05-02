'''Need to install fasttext, wget and spacy. Download "en_core_web_sm".
   Edit path and add path to WikiMatrix files. If a Wikimatrix file is missing, it will be downloaded
   
  1) Load WikiMatrix File 
  2) Go though file and create frequency dictionary of lemmas
  3) Reduce frequency dictionary to dictionary of unusual terms: lemmas with occurrences between (n=2,m=9)
  4) Go though unskipped rows in WikiMatrix file and extract valuable sentence pairs containing unusual terms

   Three .txt files should come out:
      dic_lang1-lang2.txt
      lemmatized_sentences_lang1-lang2.txt
      "Sentences_Freqs02-09_ScoreOver1.04_lang1-lang2.txt"

   If the first two are present, step 2 is skipped, so that one can re-do the analysis quickly or try another (n,m) frequency combination 
   '''


#Imports
import os
import argparse
import json
import wget

# os.chdir('../Wikimatrix') # You should change this

# Custom
from utils import *
import create_dic
import extract_goodpairs

# Gloval variables
import globals 
globals.initialize()
lang1, lang2 = globals.lang1, globals.lang2
lemmas = globals.lemmas


#Possible languages
LANGS = ['vi','ar','bs','bg','zh','hr','cs','da','nl','et','fi','fr','ka','de','el','he','hu','is','it','ja','kk','ko','lv','lt','ro','no','pl','pt','ro','ru','sr','sk','sl','es','sv','th','tr','uk','zh']

# Load English dictionary
with open('en_words.txt') as f:
    en_words = []
    for line in f:
      en_words.append(line[:-1])
    f.close()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('l', help='LANG2: second language \nPossible languages are:\n{}'.format(' '.join(LANGS)))
args = parser.parse_args()

# Define languages in alphabetical order
lang1, lang2 = sorted([str(args.l), 'en'])

# Load WikiMatrix file
path = '' # Add path to WikiMatrix files
WikiMatrix = path + 'WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2)

# If file does not exist, download it 
if not os.path.exists(WikiMatrix): 
  print('\nNo WikiMatrix file found. Downloading from net')
  #get_ipython().system('wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2)); 
  wget.download('https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2))
  WikiMatrix = 'WikiMatrix.{}-{}.tsv.gz'.format(lang1,lang2)


# Create frequency dictionary of lemmas
if os.path.exists("dic_{}-{}.txt".format(lang1,lang2)):
  dic = load_from_txt(("dic_{}-{}.txt".format(lang1,lang2)))
  lemmas = load_from_txt(("lemmatized_sentences_{}-{}.txt".format(lang1,lang2)))
  print('\nDictionary and lemmas have been loaded')
  
else: 
  print('\nNo dictionary found. Creating dictionary')
  dic, lemmas = create_dic.Run(WikiMatrix) 

  # Save dictionary and lemmas variable as .txt file
  SaveVar(data=dic, file=path+"dic_{}-{}.txt".format(lang1,lang2))
  SaveVar(data=lemmas, file=path+"lemmatized_sentences_{}-{}.txt".format(lang1,lang2))

# Reduce dictionary to lemmas with [m,n] occurrences
# Default: n, m = 2, 9
dic_red = Create_dic_red(dic)

# Extract sentences that fulfill all criteria
print('\nExtracting suitable pairs of sentences')
goodpairs = extract_goodpairs.Run(WikiMatrix, dic_red, lemmas)

# Save sentences tsv file as .txt
SaveVar(data=goodpairs, file=path+"Sentences_Freqs02-09_ScoreOver1.04_{}-{}.txt".format(lang1,lang2))
