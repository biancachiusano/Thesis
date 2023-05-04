from feature_extraction_tf import feature_extraction_tf
from text_preprocessing import text_preprocessing
from IPython.display import display
import nltk
import pandas as pd

# DATA
# 275 Violation
# 58 non violation
data = pd.read_csv('violation_or_not.csv')
violation = data.drop(data[data['Violation'] == 0].index)
violation = violation.reset_index()

non_violation = data.drop(data[data['Violation'] == 1].index)
non_violation = non_violation.reset_index()

unique_words_all = []
count_all = []
tot_all = 0

# TEXT PREPROCESSING
legal_sw = ['adjourned', 'affidavit', 'allegation', 'appeal', 'appellant', 'application', 'applicant', 'arbitration', 'case', 'cause', 'claim', 'clerk', 'complaint', 'consent', 'contempt', 'contravention', 'conviction', 'costs', 'court', 'cross-examination', 'defence', 'defendant', 'deposition', 'discovery', 'dispute', 'evidence', 'examination', 'fact', 'hearing', 'judge', 'judgment', 'jurisdiction', 'justice', 'law', 'lawsuit', 'legal', 'litigant', 'litigation', 'moot', 'motion', 'objection', 'order', 'parties', 'pleading', 'proceedings', 'ruling', 'sentence', 'settlement', 'solicitor', 'statute', 'subpoena', 'testimony', 'trial', 'verdict', 'witness']
other_sw = ['judge', 'council', 'government', 'mr', 'lawyer', 'supreme', 'judicial']


frequencies = feature_extraction_tf(unique_words_all, count_all, tot_all)
trial = ['001-223656']
#for filename in non_violation['Case']:
for filename in trial:
    trial = text_preprocessing(filename)
    cleaned = trial.clean()
    #print("CLEANED: " + cleaned)
    facts = trial.get_facts(cleaned)
    #print("FACTS: " + facts)
    processed = trial.preprocess_text(facts, legal_sw)
    #print("PROCESSED: " + processed)

    # TODO: don't know whether to pass processed or facts to the collection finder
    bigrams, trigrams, quadgrams, bigramFinder, trigramFinder, quadgramFinder = frequencies.collocations(processed)

    # Calculate scores
    print("Top 10 Bigrams with the highest PMI: ")
    frequencies.calculate_scores(bigramFinder, bigrams)
    print("Top 10 Trigrams with the highest PMI: ")
    frequencies.calculate_scores(trigramFinder, trigrams)
    print("Top 10 Quadgrams with the highest PMI: ")
    frequencies.calculate_scores(quadgramFinder, quadgrams)
    # TF-IDF

    #freq_df, unique_words_all, count_all, tot_all = frequencies.calculate_freq(processed)
    #print(len(unique_words_all))
    #print(len(count_all))
    #print(tot_all)


# PREV nn_viol_overall_df = frequencies.calculate_all_freq()
# PREV nn_viol_overall_df.to_csv('NN_VIOL_Overall_Df.csv', index=True)
#display(overall_df)
#overall_df = pd.read_csv('Overall_Df.csv')
# PREV nn_viol_groups_df = frequencies.create_groups(nn_viol_overall_df)
# PREV nn_viol_groups_df.to_csv('NN_VIOL_Groups_Df.csv')

# Name entity recognition

# Download the pre-trained NER model
#nltk.download('maxent_ne_chunker')
#nltk.download('words')

# TODO: Add more stopwords: Use the alpha words
# TODO: name entity recognition
# TODO: Stemming instead of lemmatization
# TODO: REMOVE DAYS OF THE WEEK, WEEKS, MONTHS, YEARS
# TODO: Clustering:K-means, Hierarchical clustering, density-based clustering
# TODO: Introduction and state of the arts report
# TODO: Word Embedding: WORD2VEC
