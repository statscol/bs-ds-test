
import re 
import numpy as np
import spacy
from unidecode import unidecode
import nltk
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
import pandas as pd
from config import REPLACEMENT_LANGUAGES

##stopwords from spacy 
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

nltk.download('wordnet') #in case wordnet has not been downloaded yet
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()


def to_float(text):
    try:
        value=float(text)
        return value
    except:
        return text

def remove_special_characters(sentence: str,replacement=""):
    return re.sub('[^A-Za-z0-9 ]+',replacement,sentence.lower())

def remove_multispaces(sentence: str):
    return re.sub("\s+"," ",sentence).strip()

def clean_sentence(sentence: str,lemmatizer=lemmatizer,rep=" "):
    """
    lemmatize, remove special characters, accents and standardize a sentence
    """
    sentence=unidecode(sentence)
    sentence=re.sub("([A-Z])", r' \1', sentence) # separate words by uppercase letters eg. telephoneNumber -> telephone Number
    sentence=remove_special_characters(sentence,replacement=rep)
    sentence=[lemmatizer.lemmatize(word) for word in sentence.split(" ")]  #lemmatize by noun
    sentence=[word for word in sentence if word not in stopwords or word=="not"] # remove stopword, however negation matters!
    return remove_multispaces(" ".join(sentence))

def decode_language_code(lang_code:str,decoder_dict=REPLACEMENT_LANGUAGES,sep="-"):
    lang_code=lang_code.lower().split(sep)
    lang_code=[word if word not in decoder_dict.keys() else decoder_dict[word] for word in lang_code]
    return " ".join(lang_code)

def cosine_sim(vec1,vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    
def consume_data(generator):
    try:
        return next(generator)
    except StopIteration:
        return "All batches returned"

if __name__=="__main__":
    print(clean_sentence("á,e test uk voice over"))
    print(decode_language_code("eng-us"))