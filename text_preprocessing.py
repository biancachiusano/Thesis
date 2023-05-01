import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

class text_preprocessing:
    def __init__(self, file):
        self.file = file

    def clean(self):
        with open('all_ten/' + self.file + '.txt') as f:
        #with open(self.file) as f:
            lines = f.readlines()
        filter_object = list(filter(lambda a: 'FIRST SECTION' in a, lines))
        if len(filter_object) == 0:
            filter_object = list(filter(lambda a: 'SECOND SECTION' in a, lines))
        if len(filter_object) == 0:
            filter_object = list(filter(lambda a: 'THIRD SECTION' in a, lines))
        if len(filter_object) == 0:
            filter_object = list(filter(lambda a: 'FOURTH SECTION' in a, lines))
        if len(filter_object) == 0:
            filter_object = list(filter(lambda a: 'FIFTH SECTION' in a, lines))
        # Replacing weird symbols
        cleaned = filter_object[0].replace('\xa0', ' ').replace('•', '').replace('§', '').replace('”', '')
        return cleaned


    def get_facts(self, clean):
        # removes all non-alphanumeric characters and spaces
        clean = re.sub(r'[^\w\s]', '', clean)
        # removes all digits
        clean = re.sub(r'\d+', '', clean)
        # makes sure there is a space between uppercase and lowercase text
        clean = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', clean)

        cut = clean.split()
        #print(cut)

        start_phrase = "FACTS"
        end_phrase = "LAW"
        for single in cut:
            if start_phrase in single:
                # find the start index
                start_index = cut.index(single) + len(single)
            if end_phrase in single:
                # find the end index
                end_index = cut.index(single)

        #start_index = cut.index('FACTS') + 1
        #end_index = cut.index('LAWALLEGED')

        the_facts = ' '.join(cut[start_index:end_index])
        return the_facts

    def preprocess_text(self, facts, legal_sw):

        # Tokenize the text
        tokens = word_tokenize(facts.lower())
        # Remove stop words
        all_stopwords = set(stopwords.words('english') + legal_sw)
        filtered_tokens = [token for token in tokens if token not in all_stopwords]
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
        # Join the tokens back into a string
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text



#legal_sw = ['adjourned', 'affidavit', 'allegation', 'appeal', 'appellant', 'application', 'applicant', 'arbitration', 'case', 'cause', 'claim', 'clerk', 'complaint', 'consent', 'contempt', 'contravention', 'conviction', 'costs', 'court', 'cross-examination', 'defence', 'defendant', 'deposition', 'discovery', 'dispute', 'evidence', 'examination', 'fact', 'hearing', 'judge', 'judgment', 'jurisdiction', 'justice', 'law', 'lawsuit', 'legal', 'litigant', 'litigation', 'moot', 'motion', 'objection', 'order', 'parties', 'pleading', 'proceedings', 'ruling', 'sentence', 'settlement', 'solicitor', 'statute', 'subpoena', 'testimony', 'trial', 'verdict', 'witness']
#trial = text_preprocessing('001-223656')
#cleaned = trial.clean()
#print("CLEANED: " + cleaned)
#facts = trial.get_facts(cleaned)
#print("FACTS: " + facts)
#processed = trial.preprocess_text(facts, legal_sw)
#print("PROCESSED: " + processed)

