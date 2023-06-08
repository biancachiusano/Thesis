from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


class text_preprocessing:
    def __init__(self, file):
        self.file = file

    def clean(self):
        with open('non_violation/' + self.file) as f:
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

        start_phrase = "FACTS"
        end_phrase = "LAW"
        for single in cut:
            if start_phrase in single:
                # find the start index
                start_index = cut.index(single) + len(single)
            if end_phrase in single:
                # find the end index
                end_index = cut.index(single)

        the_facts = ' '.join(cut[start_index:end_index])
        return the_facts
    # NLTK
    def preprocess_text(self, facts, legal_sw, month_sw, s_w, lemma):

        # Tokenize the text
        tokens = word_tokenize(facts.lower())
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        # Remove stop words
        all_stopwords = set(stopwords.words('english') + legal_sw + month_sw)

        # Different types of preprocessing (for experiments)
        if lemma and s_w:
            print('Both Lemma and SW')
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            # Also removing one-letter and two-letter words
            filtered_tokens = [token for token in lemmatized_tokens if token not in all_stopwords and len(token) > 2]
        if lemma and not s_w:
            print('Only Lemma')
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            filtered_tokens = lemmatized_tokens
        if s_w and not lemma:
            print('Only SW')
            filtered_tokens = [token for token in tokens if token not in all_stopwords and len(token) > 2]
        if not s_w and not lemma:
            print('No pre-processing')
            filtered_tokens = tokens
        # Join the tokens back into a string
        processed_text = ' '.join(filtered_tokens)
        return processed_text

