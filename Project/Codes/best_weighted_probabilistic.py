import pandas as pd
import numpy as np
import re
import time
from math import log10
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.book import FreqDist


class Best_Weighted_Probabilistic():

    def __init__(self):
        self.index = None
        self.docIDs = None

    def rank(self, docs, doc_ids, queries):
        for i in range(len(docs)):
            # removing weird characters
            docs[i] = re.sub(r'[^A-Za-z0-9\s]', ' ', docs[i])

        # Tokenizing the documents
        tokens = []
        for i in range(0, len(docs)):
            tokens.append(docs[i].split())
        stopwords_ = set(stopwords.words('english'))

        # Porter Stemmer is used to stem the words
        stemmer = PorterStemmer()
        final_tokens = tokens

        # looping 5 times to be sure.
        for m in range(0, 5):
            for i in range(len(final_tokens)):
                for j in range(len(final_tokens[i])):
                    if j == len(final_tokens[i]):
                        break
                    if (final_tokens[i][j] in stopwords_):
                        del final_tokens[i][j]
                    else:
                        final_tokens[i][j] = stemmer.stem(final_tokens[i][j])
                    if j == len(final_tokens[i]):
                        break

        # Create matrix of tokens
        total = 0
        for i in range(len(tokens)):
            total += len(tokens[i])
        net_tokens = []
        for i in range(0, len(final_tokens)):
            for j in range(0, len(final_tokens[i])):
                net_tokens.append(final_tokens[i][j])

        # Create a dataframe for the term frequencies
        token_freq = FreqDist(net_tokens)
        print(len(token_freq))
        final_df = pd.DataFrame(token_freq.items()).set_index(0)[1]
        # final_df.index[0]

        frequencies = pd.DataFrame(data=0, index=final_df.index, columns=range(1, 1401), dtype=np.int8)

        # Creating Term Frequencies Matrix
        count = 1
        for item in final_tokens:
            temporary = FreqDist(item)
            for i, j in temporary.items():
                frequencies.at[i, count] += j
            count += 1
        term_frequencies = pd.DataFrame(data=0, index=(final_df.index), columns=range(1, len(final_tokens) + 1),
                                        dtype=np.float16)

        for row in range(len(term_frequencies.index)):
            for col in range(1, len(final_tokens) + 1):
                if frequencies.at[frequencies.index[row], col] > 0:
                    term_frequencies.at[term_frequencies.index[row], col] = float(
                        frequencies.at[frequencies.index[row], col] / len(set(final_tokens[col - 1])))

        # Defining the Weights for Best Weighted Probability
        bwp = pd.DataFrame(data=0, index=final_df.index, columns=range(1, len(final_tokens) + 1), dtype=np.float16)
        max_tf = []
        for i in term_frequencies.index:
            max_freq = term_frequencies.at[i, 1]
            for j in range(2, len(final_tokens) + 1):
                if term_frequencies.at[i, j] > max_freq:
                    max_freq = term_frequencies.at[i, j]
            max_tf.append(max_freq)

        for i in range(len(term_frequencies.index)):
            for j in range(1, len(final_tokens) + 1):
                bwp.at[term_frequencies.index[i], j] = 0.5 + (
                            (0.5 * term_frequencies.at[term_frequencies.index[i], j]) / max_tf[i])

        # Computing the Inverse Document Frequency
        idf = pd.DataFrame(data=0, index=final_df.index, columns=['IDF', 'Count'], dtype=np.float16)
        for row in range(len(term_frequencies.index)):
            count = 0
            for col in range(1, len(final_tokens) + 1):
                if term_frequencies.at[term_frequencies.index[row], col] > 0:
                    count += 1
            if count > 0:
                idf.at[idf.index[row], 'IDF'] = log10(1400 - count / count)
                idf.at[idf.index[row], 'Count'] = count

        # Function to return the relevant documents for each query
        def query_processor(Q):
            # Removing Stopwords
            query_tokens = Q.split()
            for i in range(0, 2):
                for token in query_tokens:
                    if token in stopwords_:
                        query_tokens.remove(token)
            query_tokens.pop(len(query_tokens) - 1)

            # Stemming the query
            for i in range(len(query_tokens)):
                query_tokens[i] = stemmer.stem(query_tokens[i])
            ordering = pd.DataFrame(data=0, index=range(1, len(final_tokens) + 1), columns=['Query1'], dtype=np.float16)

            for j in range(1, len(final_tokens) + 1):
                total = 0.0
                for k in query_tokens:
                    if k in term_frequencies.index:
                        if term_frequencies.loc[k][j] != 0:
                            total += bwp.at[k, j] * idf.at[k, 'IDF']
                ordering.at[j, 'Query1'] = total
            ordering = ordering.sort_values(by=['Query1'], ascending=False)
            return ordering.index

        i = 0
        doc_IDs_ordered_BWP = []
        for query in queries:
            i = i + 1
            if (i % 10 == 0):
                print(i,' Queries completed out of 225')
                print(time.time())
            doc_IDs_ordered_BWP.append(list(query_processor(query)))
        return doc_IDs_ordered_BWP