from nltk.tokenize import TreebankWordTokenizer
from util import *
import re

class Tokenization():

    def naive(self, text):
        tokenized = []
        if isinstance(text, list):
            for s in text:
                if isinstance(s, str):
                    token = re.split(word_separators, s)
                    for word in token:
                        if ((word in punctuations) or (word ==' ') or (word =='')):
                            token.remove(word)
                    tokenized.append(token)
        else:
            print("No text received")
        return tokenized

    def pennTreeBank(self, text):
        tokenized = []
        if isinstance(text, list):
            for s in text:
                if isinstance(s, str):
                    token = TreebankWordTokenizer().tokenize(s)
                    tokenized.append(token)
        else:
            print("No text received")
        return tokenized
