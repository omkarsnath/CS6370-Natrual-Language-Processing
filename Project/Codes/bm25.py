import math
import time
from nltk.corpus import stopwords
from itertools import chain

class BM_25():

    def __init__(self):
        self.index = None
        self.docIDs = None

    def rank(self, docs, doc_ids, queries):
        texts = [list(chain.from_iterable(docs[i])) for i in range(len(docs))]

        class BM25:
            def __init__(self, k=1.85, b=0.8):
                # Best values of k and b as obtained by hyperparameter tuning
                self.b = b
                self.k = k
            def fit(self, corpus):
                term_frequency = []
                document_frequency = {}
                inv_document_frequency = {}
                documents_length_list = []
                size_of_corpus = 0

                # Processing the Documents amd creating the term frequency matrix
                for document in corpus:
                    size_of_corpus += 1
                    documents_length_list.append(len(document))
                    frequencies = {}
                    for token in document:
                        term_count = frequencies.get(token, 0) + 1
                        frequencies[token] = term_count

                    term_frequency.append(frequencies)
                    for token, _ in frequencies.items():
                        df_count = document_frequency.get(token, 0) + 1
                        document_frequency[token] = df_count

                # Creating the inverse document frequency matrix
                for token, freq in document_frequency.items():
                    inv_document_frequency[token] = math.log(1 + (size_of_corpus - freq + 0.5) / (freq + 0.5))

                # Transferring variables to global variables
                self.tf = term_frequency
                self.idf = inv_document_frequency
                self.document_length_list = documents_length_list
                self.size_corpus = size_of_corpus
                self.avg_length_list = sum(documents_length_list) / size_of_corpus
                return self

            # Function to compute scores for the given query
            def compute(self, query):
                scores = [self.compute_score(query, i) for i in range(self.size_corpus)]
                return scores

            # Function to compute score for specific query and specific document
            def compute_score(self, query, index):
                score = 0.0
                documents_length_list = self.document_length_list[index]
                frequencies = self.tf[index]
                for token in query:
                    if token not in frequencies:
                        continue
                    freq = frequencies[token]
                    num = self.idf[token] * freq * (self.k + 1)
                    den = freq + self.k * (1 - self.b + self.b * documents_length_list /self.avg_length_list)
                    score += (num / den) # The unique formula of BM25
                return score

        doc_IDs_ordered_BM25 = []
        i = 0
        # Computing and returning the relevant documents
        for query in queries:
            # Preprocessing
            query = [word for word in query.lower().split() if word not in stopwords.words('english')]

            # Creating an instance of the  class, and finding scores for the query
            bm25_instance = BM25()
            bm25_instance.fit(texts)
            scores = bm25_instance.compute(query)

            all_scores = []
            for score, doc in zip(scores, doc_ids):
                final_scores = []
                final_scores.append(score)
                final_scores.append(doc)
                all_scores.append(final_scores)

            # Ranking all the documents according to the score
            ranked_docs = sorted(all_scores, key=lambda x: x[0], reverse=True)
            doc_ordering = [ranked_docs[i][1] for i in range(1400)]
            doc_IDs_ordered_BM25.append(doc_ordering)

            # Keeping track of number of queries processed
            if (i % 10 == 0):
                print(i,' Queries completed out of 225')
                print(time.time())
            i = i + 1

        return doc_IDs_ordered_BM25