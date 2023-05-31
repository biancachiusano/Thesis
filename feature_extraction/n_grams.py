from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder, QuadgramAssocMeasures, QuadgramCollocationFinder
from nltk import word_tokenize

# https://towardsdatascience.com/generating-colocations-n-grams-6dd8bea31d13
# COLLOCATIONS: https://nlp.stanford.edu/fsnlp/promo/colloc.pdf
# https://www.nltk.org/api/nltk.collocations.html?highlight=collocations#module-nltk.collocations
class n_grams:

    def collocations(self, text):
        # collocation: pair or group of words that appear together in a specific language
        # finding collections requires first calculating the frequencies of words and their appearance in the context of other words
        # n-grams are contiguous sequences of N items from a given sample of text (bigram: pair of consecutive words in a sentence or document)
        # each ngram of words may then be scored according to some association measure, in order to determine the relative likelihood of each ngram being a collection

        # CollectionFinder: scores a ngram given appropriate frequency

        # tokenize
        tokens = word_tokenize(text)
        print(tokens)

        bigrams = BigramAssocMeasures()
        trigrams = TrigramAssocMeasures()
        quadgrams = QuadgramAssocMeasures()

        bigramFinder = BigramCollocationFinder.from_words(tokens)
        trigramFinder = TrigramCollocationFinder.from_words(tokens)
        quadgramFinder = QuadgramCollocationFinder.from_words(tokens)

        return bigrams, trigrams, quadgrams, bigramFinder, trigramFinder, quadgramFinder

    def calculate_scores(self, finder, measures):
        pmi = finder.score_ngrams(measures.pmi)

        # Print the top 10 bigrams with the highest PMI
        for gram in pmi[:10]:
            print(gram)