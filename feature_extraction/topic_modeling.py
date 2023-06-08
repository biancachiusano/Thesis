import pandas as pd

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim

#import spacy

class topic_modeling:
    def __init__(self, facts):
        self.facts = facts

    def organise(self):
        facts_words = []
        for fact in self.facts:
            split_fact = fact.split()
            facts_words.append(split_fact)
        return facts_words





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
