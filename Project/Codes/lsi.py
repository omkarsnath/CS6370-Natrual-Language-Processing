import numpy as np
import re
import time
from itertools import chain
import scipy.sparse as sp
from scipy.sparse.linalg import svds
pattern = re.compile(r'\W+')

class LSI():

    def __init__(self):
        self.index = None
        self.docIDs = None

    def rank(self, docs, doc_ids, queries):
        z = 800  # this value for the rank of the matrix has been chosen after doing a hyperparameter tuning analysis
        # based on the maximum value of the MAP for all the queries
        
        # Converting all words to lower case
        terms_no = 1
        vocabulary = dict()
        for i in range(len(docs)):
            for word in list(chain.from_iterable(docs[i])):
                lower_word = word.lower()
                if lower_word in vocabulary:
                    continue
                else:
                    vocabulary[lower_word] = terms_no
                    terms_no += 1
        terms_no -= 1
        
        row = []
        col = []
        frequency = []
        # Function to create the true frequency matrix
        def true_frequency(idx):
            tf_dict = dict()
            for word in list(chain.from_iterable(docs[i])):
                word = word.lower()
                if word in tf_dict:
                    tf_dict[word] += 1.0
                else:
                    tf_dict[word] = 1.0
            for (term, freq) in tf_dict.items():
                row.append(vocabulary[term] - 1)
                col.append(idx)
                frequency.append(freq)

        # Creating the true frequency matrix
        for i in range(0, len(docs)):  # iterate over all documents
            true_frequency(i)

        # Performing SVD of the given matrix
        tf_matrix = sp.csc_matrix((frequency, (row, col)), shape=(terms_no, len(docs)))
        num_docs = 1400 # number of documents
        U, S, Vt = svds(tf_matrix, k=z, which='LM')  # U - nxk; V - kxm
        V = Vt.T
        threshold = 0.000000000000001
        sinv = []
        for t in S:
            if t < threshold:
                sinv.append(0.0)
            else:
                sinv.append(1.0 / t)

        sinv = np.array(sinv)
        sinv = np.diag(sinv)
        S = np.diag(S)
        VS = np.dot(V, S)
        doc_IDs_ordered_LSI = []
        tmp = np.dot(U, sinv)

        # Processing all the queries
        j=0
        for query in queries:
            ranked_docs = []

            # Preprocessing the query
            tf_vector = [0] * len(vocabulary)
            query = query.lower()
            tokens = pattern.split(query)
            for token in tokens:
                token = token.lower()
                if token in vocabulary:
                    tf_vector[vocabulary[token] - 1] += 1

            # Reducing the tf vector
            D = np.dot(tf_vector, tmp)
            N = np.linalg.norm(D)

            # Computing the similarity for the query and ranking the documents
            similarity = []
            for i, R in enumerate(VS):
                val = np.dot(R, D) / (np.linalg.norm(R) * N)
                similarity.append((val, i + 1))
            similarity.sort(key=lambda x: -x[0])
            for val, i in similarity[:num_docs + 1]:
                ranked_docs.append(doc_ids[i - 1])
            doc_IDs_ordered_LSI.append(ranked_docs)

            if (j % 10 == 0):
                print(j,' Queries completed out of 225')
                print(time.time())
            j=j+1

        return doc_IDs_ordered_LSI