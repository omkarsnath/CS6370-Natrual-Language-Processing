from util import *
from nltk.tokenize import PunktSentenceTokenizer
import re
import nltk
nltk.download('punkt')

class SentenceSegmentation():

    def naive(self, text):
        if isinstance(text, str):
            segments = re.split(delimiters, text)
            sentences = [s.strip() for s in segments]
            while '' in sentences:
                sentences.remove('')
            return sentences
        else:
            print("No text received")
            return ([])

    def punkt(self, text):
        if (isinstance(text, str)):
            tokenizer = PunktSentenceTokenizer(text)
            sentences = tokenizer.tokenize(text)
            return sentences
        else:
            print("No text received")
            return ([])

