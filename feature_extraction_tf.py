from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder, QuadgramAssocMeasures, QuadgramCollocationFinder
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize

# COLLOCATIONS: https://nlp.stanford.edu/fsnlp/promo/colloc.pdf
# https://www.nltk.org/api/nltk.collocations.html?highlight=collocations#module-nltk.collocations
# https://towardsdatascience.com/text-clustering-using-k-means-ec19768aae48
# https://towardsdatascience.com/generating-colocations-n-grams-6dd8bea31d13
#https://towardsdatascience.com/nlp-for-topic-modeling-summarization-of-legal-documents-8c89393b1534



class feature_extraction_tf:
    def __init__(self, unique_words_all, count_all, tot_all):
        #self.file = file
        self.unique_words_all = unique_words_all
        self.count_all = count_all
        self.tot_all = tot_all

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

    #def feature_matrix(self, score):


    def calculate_freq(self, file):
        # For every word in the text
        # if word has not been seen, but in list

        unique_words = []
        count_list = []
        #frequency_list = []

        freq_df = pd.DataFrame(columns=['Words', 'Count', 'T_Freq'])

        list_words = file.split()
        tot_words = len(list_words)  # 1075

        # COUNT TOTAL NUMBER OF WORDS IN ALL DOCUMENTS
        self.tot_all = self.tot_all + tot_words

        # CHECK UNIQUE WORDS IN THE DOC
        for word in list_words:
            if word not in unique_words:
                unique_words.append(word)


        # NUMBER OF OCCURRENCES OF SPECIFIC WORD IN DOC
        for unique in unique_words:
            count = 0
            for word in list_words:
                if unique == word:
                    count = count + 1
            count_list.append(count)


        freq_df['Words'] = unique_words  # 541
        freq_df['Count'] = count_list

        # CHECK UNIQUE WORDS OVERALL (if there are any new unique words in this doc)
        for new_word in unique_words:
            index = unique_words.index(new_word)
            if new_word not in self.unique_words_all:
                #print(new_word)
                self.unique_words_all.append(new_word)
                # add count
                self.count_all.append(count_list[index])
                #print(count_list[index])
            else:
                # get index of position of existing word in unique_words_all
                existing = self.unique_words_all.index(new_word)
                current_count = self.count_all[existing]
                new_count = current_count + count_list[index]
                self.count_all[existing] = new_count

        # CALCULATE FREQUENCY
        freq_df['T_Freq'] = freq_df['Count'] / tot_words
        freq_df = freq_df.sort_values('T_Freq', ascending=False).reset_index(drop=True)

        return freq_df, self.unique_words_all, self.count_all, self.tot_all


    def calculate_all_freq(self):
        all_freqs= pd.DataFrame(columns=['Words', 'Count', 'T_Freq'])
        all_freqs['Words'] = self.unique_words_all
        all_freqs['Count'] = self.count_all

        # CALCULATE FREQUENCY
        all_freqs['T_Freq'] = all_freqs['Count'] / self.tot_all
        all_freqs = all_freqs.sort_values('T_Freq', ascending=False).reset_index(drop=True)
        return all_freqs

    def create_groups(self, table):
        all_words = table.iloc[:, 0].tolist()
        only_str = []
        print(all_words)
        keep_track = []
        for entry in all_words:
            if isinstance(entry, str):
                only_str.append(entry)

        for i in range(101):
            entry = only_str[i]
            if entry not in keep_track:
                new_col = []
                keep_track.append(entry)
                print(entry)
                matches = [word for word in only_str if entry in word]
                if len(matches) > 0:
                    new_col.append(entry)
                    for match in matches:
                        keep_track.append(match)
                        new_col.append(match)
                    table[str(entry)] = pd.Series(new_col, index=table.index[:len(new_col)])
        return table

    def topic_modelling(self, text):
        # CountVectorize: number of times a word occurs in the list -> Document Term Matrix
        # TODO: change ngram_range
        vectorizer = CountVectorizer(ngram_range=(1,1))
        # TODO: I need to create documents with only the final text so it can iterate over documents
        dtm = vectorizer.fit_transform(text)
        pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names_out())
        lda = LatentDirichletAllocation(n_components=5)
        lda.fit_transform(dtm)
        #sorting = np.argsort(lda.components_)[:,::-1]
        #features = np.array(vectorizer.get_feature_names_out())
        #mglearn.tools.print_topics(topics=range(5), feature_names=features, sorting=sorting, topics_per_chunk=5, n_words=10)






