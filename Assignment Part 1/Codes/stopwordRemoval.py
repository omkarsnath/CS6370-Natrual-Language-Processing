from util import *
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

class StopwordRemoval():

    def fromList(self, text):
        for s in range(len(text)):
                token_words = []
                tokens = text[s]
                for word in tokens:
                        if word not in stop_words:
                                token_words.append(word)
                text[s] = token_words
        return text

