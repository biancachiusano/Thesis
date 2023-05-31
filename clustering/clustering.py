

'''

K-means clustering
A method of vector quantization
Unsupervised algorithms
Partitions n observations into k clusters in which each ovsevation belongs to the cluster with the nearest mean
NP hard - so heuristic algorithms
Algorithms: naive k means

https://towardsdatascience.com/text-clustering-using-k-means-ec19768aae48

https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7

https://www.youtube.com/watch?v=iNlZ3IU5Ffw


K_means (do also the optimal ks method and experiment with different ks)
DBSCAN clustering
Clustering with tf-idf
'''

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# https://towardsdatascience.com/text-clustering-using-k-means-ec19768aae48

class clustering:

    def vectorise(self, all_text):
        # passing in a list with all the processed facts
        vectorizer = CountVectorizer()
        matrix = vectorizer.fit_transform(all_text)
        count_vect = pd.DataFrame(data=matrix.toarray(), columns=vectorizer.get_feature_names_out())
        return count_vect

    def k_means(self, all_text):
        # Transform to dataframe
        facts_df = pd.DataFrame(columns=['Facts','Cluster', 'x0', 'x1'])
        facts_df['Facts'] = all_text

        '''
           # Vectorise
        vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
        # Matrix rray of vectors that will be used to train the KMeans model (sparse matrix)
        matrix = vectorizer.fit_transform(facts_df['Facts'])

        # K means
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(matrix)
        clusters = kmeans.labels_

        # Visualisation
        pca = PCA(n_components=2, random_state=42)
        pca_vecs = pca.fit_transform(matrix.toarray())
        x0 = pca_vecs[:, 0]
        x1 = pca_vecs[:, 1]
        facts_df['Cluster'] = clusters
        facts_df['x0'] = x0
        facts_df['x1'] = x1

        # getting the top keywords
        top_words = pd.DataFrame(matrix.todense()).groupby(clusters).mean()
        terms = vectorizer.get_feature_names_out()
        for i,r in top_words.iterrows():
            print('\nCluster {}'.format(i))
            print(','.join([terms[t] for t in np.argsort(r)[-10:]]))

        cluster_map = {0: 'imprisonment', 1: 'doc_mur_ab', 2: 'Writing'}
        facts_df['Cluster'] = facts_df['Cluster'].map(cluster_map)
       
        '''

        '''
        # set image size
        plt.figure(figsize=(12, 7))
        # set a title
        plt.title("TF-IDF + KMeans 20newsgroup clustering", fontdict={"fontsize": 18})
        # set axes names
        plt.xlabel("X0", fontdict={"fontsize": 16})
        plt.ylabel("X1", fontdict={"fontsize": 16})
        # create scatter plot with seaborn, where hue is the class used to group the data
        sns.scatterplot(data=facts_df, x='x0', y='x1', hue='Cluster', palette="viridis")
        plt.show()
        '''
        return facts_df



