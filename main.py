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

unique_words_all = []
count_all = []
tot_all = 0

# TEXT PREPROCESSING
legal_sw = ['adjourned', 'affidavit', 'allegation', 'appeal', 'appellant', 'application', 'applicant', 'arbitration', 'case', 'cause', 'claim', 'clerk', 'complaint', 'consent', 'contempt', 'contravention', 'conviction', 'costs', 'court', 'cross-examination', 'defence', 'defendant', 'deposition', 'discovery', 'dispute', 'evidence', 'examination', 'fact', 'hearing', 'judge', 'judgment', 'jurisdiction', 'justice', 'law', 'lawsuit', 'legal', 'litigant', 'litigation', 'moot', 'motion', 'objection', 'order', 'parties', 'pleading', 'proceedings', 'ruling', 'sentence', 'settlement', 'solicitor', 'statute', 'subpoena', 'testimony', 'trial', 'verdict', 'witness']
trial_violation = ['001-223656', '001-211828']
trial_violation = ['001-160270']

frequencies = feature_extraction_tf(unique_words_all, count_all, tot_all)

for filename in violation['Case']:
#for filename in trial_violation:
    trial = text_preprocessing(filename)
    cleaned = trial.clean()
    #print("CLEANED: " + cleaned)
    facts = trial.get_facts(cleaned)
    #print("FACTS: " + facts)
    processed = trial.preprocess_text(facts, legal_sw)
    #print("PROCESSED: " + processed)

    # TF-IDF

    freq_df, unique_words_all, count_all, tot_all = frequencies.calculate_freq(processed)
    print(len(unique_words_all))
    print(len(count_all))
    print(tot_all)

    #frequencies = feature_extraction_tf(processed)
    #freq_df = frequencies.calculate_freq()
    #display(freq_df)

overall_df = frequencies.calculate_all_freq()
#overall_df.to_csv('Overall_Df.csv', index=True)
display(overall_df)

# Name entity recognition

# Download the pre-trained NER model
#nltk.download('maxent_ne_chunker')
#nltk.download('words')