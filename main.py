from feature_extraction_tf import feature_extraction_tf
from text_preprocessing import text_preprocessing
from IPython.display import display

# TEXT PREPROCESSING
legal_sw = ['adjourned', 'affidavit', 'allegation', 'appeal', 'appellant', 'application', 'applicant', 'arbitration', 'case', 'cause', 'claim', 'clerk', 'complaint', 'consent', 'contempt', 'contravention', 'conviction', 'costs', 'court', 'cross-examination', 'defence', 'defendant', 'deposition', 'discovery', 'dispute', 'evidence', 'examination', 'fact', 'hearing', 'judge', 'judgment', 'jurisdiction', 'justice', 'law', 'lawsuit', 'legal', 'litigant', 'litigation', 'moot', 'motion', 'objection', 'order', 'parties', 'pleading', 'proceedings', 'ruling', 'sentence', 'settlement', 'solicitor', 'statute', 'subpoena', 'testimony', 'trial', 'verdict', 'witness']
trial = text_preprocessing('001-223656')
cleaned = trial.clean()
print("CLEANED: " + cleaned)
facts = trial.get_facts(cleaned)
print("FACTS: " + facts)
processed = trial.preprocess_text(facts, legal_sw)
print("PROCESSED: " + processed)

# TF-IDF
frequencies = feature_extraction_tf(processed)
freq_df = frequencies.calculate_freq()
display(freq_df)