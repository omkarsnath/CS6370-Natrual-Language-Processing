from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
import argparse
import json
from sys import version_info

if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")

class SearchEngine:
    def __init__(self, args):
        self.args = args
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()

    def segmentSentences(self, text):
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        return self.stopwordRemover.fromList(text)

    def preprocessQueries(self, queries):
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(
            self.args.out_folder + "segmented_queries.txt", 'w'))
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(
            self.args.out_folder + "tokenized_queries.txt", 'w'))
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(
            self.args.out_folder + "reduced_queries.txt", 'w'))
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(
            self.args.out_folder + "stopword_removed_queries.txt", 'w'))

        preprocessedQueries = stopwordRemovedQueries
        return preprocessedQueries

    def preprocessDocs(self, docs):
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(
            self.args.out_folder + "segmented_docs.txt", 'w'))
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(
            self.args.out_folder + "tokenized_docs.txt", 'w'))
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(
            self.args.out_folder + "reduced_docs.txt", 'w'))
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(
            self.args.out_folder + "stopword_removed_docs.txt", 'w'))

        preprocessedDocs = stopwordRemovedDocs
        return preprocessedDocs

    def evaluateDataset(self):
        queries_json = json.load(
            open(args.dataset + "cran_queries.json", 'r'))[:]
        queries = [item["query"] for item in queries_json]
        processedQueries = self.preprocessQueries(queries)
        docs_json = json.load(
            open(args.dataset + "cran_docs.json", 'r'))[:][:4]
        docs = [item["body"] for item in docs_json]
        processedDocs = self.preprocessDocs(docs)

    def handleCustomQuery(self):
        print("Enter query below")
        query = input()
        processedQuery = self.preprocessQueries([query])[0]

        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:10]
        docs = [item["body"] for item in docs_json]
        processedDocs = self.preprocessDocs(docs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main.py')

    parser.add_argument('-dataset', default="cranfield/",
                        help="Path to the dataset folder")
    parser.add_argument('-out_folder', default="output/",
                        help="Path to output folder")
    parser.add_argument('-segmenter', default="punkt",
                        help="Sentence Segmenter Type [naive|punkt]")
    parser.add_argument('-tokenizer',  default="ptb",
                        help="Tokenizer Type [naive|ptb]")
    parser.add_argument('-custom', action="store_true",
                        help="Take custom query as input")

    args = parser.parse_args()

    searchEngine = SearchEngine(args)

    if args.custom:
        searchEngine.handleCustomQuery()
    else:
        searchEngine.evaluateDataset()
