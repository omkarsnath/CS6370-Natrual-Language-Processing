# Add your import statements here
from nltk.corpus import wordnet
punctuations = ['\'','\"','?', ':', '!', '.', ',', ';','&','#','(',')','[',']','{','}','_','|']
delimiters = '[.?!]'
word_separators = "[' ,-/]"

def convert_to_wordnet(s):
    if s.startswith("J"):
        return wordnet.ADJ
    elif s.startswith("V"):
        return wordnet.VERB
    elif (s.startswith("N")) | (s.startswith("P")):
        return wordnet.NOUN
    elif s.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Add any utility functions here
