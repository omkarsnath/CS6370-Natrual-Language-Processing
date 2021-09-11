from util import *
from collections import Counter
import math


class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.docIDs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = {}

		for doc in docs:
			doc_ID = docIDs[docs.index(doc)]
			all_terms = [term for sentence in doc for term in sentence]
			for term, frequency in list(Counter(all_terms).items()):
				try:
					index[term].append([doc_ID, frequency])
				except:
					index[term] = [[doc_ID,frequency]]

		self.docIDs = docIDs			
		self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		index = self.index
		docIDs = self.docIDs

		inv_frequency = {}
		null = {}
		num_docs = len(docIDs)

		for term in index:
			num_terms = len(index[term])
			inv_frequency[term] = math.log10(float(num_docs/num_terms))
			null[term] = 0

		# Representing in tf-idf vector space
		documents = {}
		for doc_ID in docIDs:
			documents[doc_ID] = null.copy()

		for term in index:
			for doc_ID, frequency in index[term]:
				documents[doc_ID][term] = frequency * inv_frequency[term]

		# Representing queries in tf-idf vector space
		for query in queries:
			query_vector = null.copy()
			terms = [term for sentence in query for term in sentence]

			for term, frequency in list(Counter(terms).items()):
				try:
					query_vector[term] = frequency * inv_frequency[term]
				except:
					pass

			similarities = {}
			for doc_ID in docIDs:
				try:
					similarities[doc_ID] = sum(documents[doc_ID][key] * query_vector[key] for key in index) / (math.sqrt(sum(documents[doc_ID][key] * documents[doc_ID][key] for key in index)) * math.sqrt(sum(query_vector[key] * query_vector[key] for key in index)))
				except:
					similarities[doc_ID] = 0
			doc_IDs_ordered.append([docID for docID, tf in sorted(similarities.items(), key=lambda item: item[1], reverse = True)])

		return doc_IDs_ordered