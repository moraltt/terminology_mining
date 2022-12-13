import pandas as pd
from xml.dom import minidom
import time
import gzip
from bs4 import BeautifulSoup
import re

def process_tuv(tuv):
    # Extract language and sentence
    lang = tuv.getAttribute("lang")
    if lang == '':
        lang = tuv.getAttribute("xml:lang")
    seg = tuv.getElementsByTagName('seg')[0]
    txt = seg.childNodes[0].data
    return lang, txt

def read_tmx(path):

    """Takes TMX translation memory file and outputs the tmx metadata and a pandas dataframe.
    Args:
        param1 (str): The path to the TMX translation file
    Returns:
        dict: The header of the TMX file, which contains metadata
        DataFrame: A Pandas Dataframe. Each line item consists of source_language, source_sentence, target_language, target_sentence,
        and other possible metadata
    """
    start_time = time.time()
    # parse an xml file by name
    tmx = minidom.parse(path)

    # Get full tmx metadata
    metadata = {}
    header = tmx.getElementsByTagName('header')[0]
    for key in header.attributes.keys():
        metadata[key] = header.attributes[key].value
        
    srclang = metadata['srclang']

    # Get translation sentences
    body = tmx.getElementsByTagName('body')[0]
    translation_units = body.getElementsByTagName('tu')
    items = []
    count_unpaired = 0
    for tu in translation_units:
        if len(tu.getElementsByTagName('tuv')) < 2: # Skip lines lacking translation 
            count_unpaired = count_unpaired + 1 
        else:
            try:
                srclang, srcsentence = process_tuv(tu.getElementsByTagName('tuv')[0]) # Source
                targetlang, targetsentence = process_tuv(tu.getElementsByTagName('tuv')[1]) # Target
                item = {
                    'source_language': srclang,
                    'source_sentence': srcsentence,
                    'target_language': targetlang,
                    'target_sentence': targetsentence
                }
                meta = {x[0]:x[1] for x in tu.attributes.items()} # Get sentence's metadata
                
                items.append({**item,**meta}) # Populate 
    
            except: count_unpaired = count_unpaired + 1; continue # Skip lines delivering error (incorrect format) 

    df = pd.DataFrame(items) # Create dataframe
    if count_unpaired > 0:
       print("The data contained %d problematic lines which were ignored" % (count_unpaired))
    
    print("Corpus created in %.3f seconds with %d elements" %((time.time()-start_time),len(df)))
    return metadata, df


def standarize(txt):
    txt=re.sub('\t|\n', ' ',txt)
    txt=re.sub(' +', ' ',txt)
    return txt.strip()

def process_tuv_2(tuv):
    # Extract language and sentence
    try: lang = tuv['lang']
    except: lang = tuv['xml:lang']
    txt = tuv.seg.text
    return lang, standarize(txt)

def read_tmx_2(path):
    # Second version, using BeautifulSoup
    """Takes TMX translation memory file and outputs the tmx metadata and a pandas dataframe.
    Args:
        param1 (str): The path to the TMX translation file
    Returns:
        dict: The header of the TMX file, which contains metadata
        DataFrame: A Pandas Dataframe. Each line item consists of source_language, source_sentence, target_language, target_sentence,
        and other possible metadata
    """
    start_time = time.time()
    # parse an xml file by name
    tmx = BeautifulSoup(path, features="lxml-xml")

    # Get full tmx metadata
    metadata = tmx.header.attrs

    # Get translation sentences
    translation_units = tmx.find_all('tu')
    items = []
    count_unpaired = 0
    for tu in translation_units:
        if len(tu.find_all('tuv')) < 2: # Skip lines lacking translation 
            count_unpaired = count_unpaired + 1 
        else:
            try:
                srclang, srcsentence = process_tuv_2(tu.find_all('tuv')[0]) # Source
                targetlang, targetsentence = process_tuv_2(tu.find_all('tuv')[1]) # Target
                item = {
                    'source_language': srclang,
                    'source_sentence': srcsentence,
                    'target_language': targetlang,
                    'target_sentence': targetsentence
                }
                meta = tu.attrs # Get sentence's metadata
                
                items.append({**item,**meta}) # Populate 
    
            except: count_unpaired = count_unpaired + 1; continue # Skip lines delivering error (incorrect format) 

    df = pd.DataFrame(items) # Create dataframe
    if count_unpaired > 0:
        print("The data contained %d problematic lines which were ignored" % (count_unpaired))
    
    print("Corpus created in %.3f seconds with %d elements" %((time.time()-start_time),len(df)))
    return metadata, df
 
