import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from gensim.models import TfidfModel
import pyLDAvis
import pyLDAvis.gensim


class topic_modeling:
    def __init__(self, facts):
        self.facts = facts

    def organise(self):
        facts_words = []
        for fact in self.facts:
            split_fact = fact.split()
            facts_words.append(split_fact)
        return facts_words

    def bi_grams(self, data):
        bigram_phrases = gensim.models.Phrases(data, min_count=5, threshold=100)
        bigram = gensim.models.phrases.Phraser(bigram_phrases)

        return (bigram[doc] for doc in data), bigram_phrases, bigram

    def tri_grams(self, data, data_bigram, bigram_phrases, bigram):
        trigram_phrases = gensim.models.Phrases(bigram_phrases[data], threshold=100)
        trigram = gensim.models.phrases.Phraser(trigram_phrases)

        return (trigram[bigram[doc]] for doc in data_bigram)

    def tf_idf(self, data_bigrams_trigrams):

        id2word = corpora.Dictionary(data_bigrams_trigrams)

        texts = data_bigrams_trigrams

        corpus = [id2word.doc2bow(text) for text in texts]
        # print (corpus[0][0:20])

        tfidf = TfidfModel(corpus, id2word=id2word)

        low_value = 0.03
        words = []
        words_missing_in_tfidf = []
        for i in range(0, len(corpus)):
            bow = corpus[i]
            low_value_words = []  # reinitialize to be safe. You can skip this.
            tfidf_ids = [id for id, value in tfidf[bow]]
            bow_ids = [id for id, value in bow]
            low_value_words = [id for id, value in tfidf[bow] if value < low_value]
            drops = low_value_words + words_missing_in_tfidf
            for item in drops:
                words.append(id2word[item])
            words_missing_in_tfidf = [id for id in bow_ids if
                                      id not in tfidf_ids]  # The words with tf-idf socre 0 will be missing

            new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
            corpus[i] = new_bow

        return corpus, id2word

    def perform_lda(self, corpus, id2word):
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=10,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=100,
                                                    passes=10,
                                                    alpha="auto")

        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        sns.set_style("white")
        pyLDAvis.display(vis)




'''
violation_facts = pd.read_csv('violation_csv/violation_facts_df_for_cluster.csv')
facts = violation_facts['Facts'].tolist()
tm = topic_modeling(facts)
facts_words = tm.organise()
print(len(facts_words))
print(facts_words[0])
lemmatized_texts = tm.lemmatization(facts_words[0])
print(lemmatized_texts)

'''
