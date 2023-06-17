import gensim
from gensim.models import CoherenceModel
import pandas as pd
import nltk
from nltk import word_tokenize

class topic_modeling:
    def __init__(self, facts):
        self.facts = facts

    def organise(self):
        facts_words = []
        for fact in self.facts:
            split_fact = fact.split()
            facts_words.append(split_fact)
        return facts_words

    def remove_lang_words(self,facts_words, english_words):
        # Filtering out non-english words and three letter words
        filtered = [word for word in facts_words if word in english_words and len(word)>3]
        return filtered

    def find_bigrams(self, data):

        bigram_list = []

        for facts in data:
            text = ' '.join(facts)
            tokens = word_tokenize(text)
            bigrams = list(nltk.bigrams(tokens))
            bigram_list.append(bigrams)

        combined_tuples = []
        final_bigram_list = []
        for bigrams in bigram_list:
            combined_tuples = []
            for tuples in bigrams:
                combined = '_'.join(tuples)
                combined_tuples.append(combined)
            final_bigram_list.append(combined_tuples)

        return final_bigram_list

    def view_frequencies(self, id2word, texts, corpus, tfidf):
        word_frequency = []
        for doc in corpus:
            # Convert the document to TF-IDF representation
            tfidf_vector = tfidf[doc]

            #Iterate over the TF-IDF values for each word in the document
            for word_id, tfidf_score in tfidf_vector:
                # Get the word corresponding to the word ID
                word = id2word[word_id]
                word_frequency.append([word,tfidf_score])

        df_freq = pd.DataFrame(word_frequency, columns = ['Word', 'Frequency'])
        df_sorted = df_freq.sort_values('Frequency', ascending=False)

        return df_sorted

    def filter_tf_idf(self, id2word, texts, corpus, tfidf):

        low_value = 0.03
        high_value = 0.6
        words = []
        words_missing_in_tfidf = []
        for i in range(0, len(corpus)):
            bow = corpus[i]
            tfidf_ids = [id for id, value in tfidf[bow]]
            bow_ids = [id for id, value in bow]
            value_words = [id for id, value in tfidf[bow] if value < low_value or value > high_value]
            drops = value_words + words_missing_in_tfidf

            for item in drops:
                words.append(id2word[item])
            words_missing_in_tfidf = [id for id in bow_ids if
                                      id not in tfidf_ids]  # The words with tf-idf socre 0 will be missing

            new_bow = [b for b in bow if b[0] not in value_words and b[0] not in words_missing_in_tfidf]
            corpus[i] = new_bow

        return corpus, id2word

    def perform_lda(self, corpus, id2word):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=10,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=50,
                                                    passes=10,
                                                    alpha="auto")

        return lda_model

    def show_results(self, lda_model):

        topics_df = pd.DataFrame(columns=['Topic', 'Top 5 Terms', 'Prob'])
        results = lda_model.show_topics(num_topics=10, num_words=5, log=False, formatted=False)

        for i in range(0, 10):
            terms_five = []
            prob_five = []
            for j in range(0, 5):
                terms_five.append(results[i][1][j][0])
                prob_five.append(results[i][1][j][1])

            new_row = pd.DataFrame({'Topic': [i], 'Top 5 Terms': [terms_five], 'Prob': [prob_five]})
            topics_df = pd.concat([topics_df, new_row], ignore_index=True)

        return topics_df