from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag
from util import *
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class InflectionReduction:
    def reduce(self, text):
        if isinstance(text, list):
            for s in range(len(text)):

                while '' in text[s]:
                    text[s].remove('')
                pos_tags = pos_tag(text[s])

                for word in range(len(text[s])):
                    pos = convert_to_wordnet(pos_tags[word][1])
                    text[s][word] = wordnet_lemmatizer.lemmatize(text[s][word], pos=pos)
            return text
        else:
            print("No text received")
            return []

