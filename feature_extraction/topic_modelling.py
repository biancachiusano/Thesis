from sklearn.decomposition import LatentDirichletAllocation
#https://towardsdatascience.com/nlp-for-topic-modeling-summarization-of-legal-documents-8c89393b1534

class topic_modelling:

    def topic_modelling(self, dtm):
        # CountVectorize: number of times a word occurs in the list -> Document Term Matrix
        lda = LatentDirichletAllocation(n_components=5)
        lda_dtf= lda.fit_transform(dtm)
        print(lda_dtf)
        #sorting = np.argsort(lda.components_)[:,::-1]
        #features = np.array(vectorizer.get_feature_names_out())
        #mglearn.tools.print_topics(topics=range(5), feature_names=features, sorting=sorting, topics_per_chunk=5, n_words=10)
