from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re


class text_preprocessing:
    def __init__(self, file):
        self.file = file

    def clean(self):
        with open('violation/' + self.file) as f:
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

    def preprocess_text(self, facts, legal_sw, month_sw):

        # Tokenize the text
        tokens = word_tokenize(facts.lower())
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Remove stop words
        all_stopwords = set(stopwords.words('english') + legal_sw + month_sw)
        # Also removing one-letter words
        filtered_tokens = [token for token in lemmatized_tokens if token not in all_stopwords and len(token) > 1]
        # Join the tokens back into a string
        processed_text = ' '.join(filtered_tokens)
        return processed_text