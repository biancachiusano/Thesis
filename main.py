import os

from clustering import clustering
from feature_extraction_tf import feature_extraction_tf
from text_preprocessing import text_preprocessing
from IPython.display import display
import nltk
import pandas as pd
import non_violation
import violation

# DATA
# 275 Violation
# 74 Non-Violation Documents

final_non_violation = pd.read_csv('final_non_violation.csv')
final_non_violation = final_non_violation.reset_index()

final_violation = pd.read_csv('final_violation.csv')
final_violation = final_violation.reset_index()


all_processed_facts = []
unique_words_all = []
count_all = []
tot_all = 0

# TEXT PREPROCESSING
legal_sw = ['adjourned', 'affidavit', 'allegation', 'appeal', 'appellant', 'application', 'applicant', "applicant's",
            'arbitration','case', 'cause', 'claim', 'clerk', 'complaint', 'consent', 'contempt', 'contravention',
            'conviction','costs', 'court', 'cross-examination', 'defence', 'defendant', 'deposition', 'discovery',
            'dispute','evidence', 'examination', 'fact', 'hearing', 'judge', 'judgment', 'jurisdiction', 'justice','law',
            'lawsuit', 'legal', 'litigant', 'litigation', 'moot', 'motion', 'objection', 'order', 'parties', 'pleading',
            'proceedings', 'ruling', 'sentence', 'settlement', 'solicitor', 'statute', 'subpoena', 'testimony', 'trial',
            'verdict', 'witness', 'cases', 'courts', "litigant's", "defendant's","judge's", 'council', 'government',
            'mr', 'lawyer', 'supreme', 'judicial', 'ha', 'wa']
month_sw = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november'
            , 'december']

frequencies = feature_extraction_tf(unique_words_all, count_all, tot_all)
clusters = clustering()
practice_files = ['001-210075.txt']

#for filename in final_non_violation['Case']:
for filename in final_violation['Case']:
#for filename in practice_files:
    trial = text_preprocessing(filename)
    cleaned = trial.clean()
    # print("CLEANED: " + cleaned)
    facts = trial.get_facts(cleaned)
    # print("FACTS: " + facts)
    processed = trial.preprocess_text(facts, legal_sw, month_sw)
    # print("PROCESSED: " + processed)


    '''
    # TODO: don't know whether to pass processed or facts to the collection finder
    bigrams, trigrams, quadgrams, bigramFinder, trigramFinder, quadgramFinder = frequencies.collocations(facts)
    
    
    print("Top 10 Bigrams with the highest PMI: ")
    frequencies.calculate_scores(bigramFinder, bigrams)
    print("Top 10 Trigrams with the highest PMI: ")
    frequencies.calculate_scores(trigramFinder, trigrams)
    print("Top 10 Quadgrams with the highest PMI: ")
    frequencies.calculate_scores(quadgramFinder, quadgrams)
    
    
    # AFTER HAVING REMOVED STOP WORDS and DONE LEMMATIZER
    bigrams, trigrams, quadgrams, bigramFinder, trigramFinder, quadgramFinder = frequencies.collocations(processed)
    
    # Calculate scores
    print("Top 10 Bigrams with the highest PMI: ")
    frequencies.calculate_scores(bigramFinder, bigrams)
    print("Top 10 Trigrams with the highest PMI: ")
    frequencies.calculate_scores(trigramFinder, trigrams)
    print("Top 10 Quadgrams with the highest PMI: ")
    frequencies.calculate_scores(quadgramFinder, quadgrams)
    
    '''
    # TF-IDF
    freq_df, unique_words_all, count_all, tot_all = frequencies.calculate_freq(processed)
    # frequencies.topic_modelling(processed)
    # DEBUGGING
    # print(len(unique_words_all))
    # print(len(count_all))
    # print(tot_all)

    # Save all processed facts in a list
    all_processed_facts.append(processed)

# CSVs
viol_overall_df = frequencies.calculate_all_freq()
viol_overall_df.to_csv('violation_overall.csv', index=True)

viol_groups_df = frequencies.create_groups(viol_overall_df)
viol_groups_df.to_csv('violation_groups.csv')

# VECTORISE
#count_vect_df = clusters.vectorise(all_processed_facts)
#print(count_vect_df)
#print(count_vect_df.shape)
#frequencies.topic_modelling(count_vect_df)

# CLUSTERING K-MEANS BASED ON TF_IDF
violation_facts_df = clusters.k_means(all_text=all_processed_facts)
violation_facts_df.to_csv('violation_facts_df_for_cluster.csv')



# TODO: Name entity recognition
# TODO: should I remove words that are not in english?

'''

print(len(files_to_consider))
print(files_to_consider)

zeros = 0
non_zeros = 0
final_violation = []
for entry in files_to_consider:
    if entry == 0:
        zeros = zeros + 1
    else:
        non_zeros = non_zeros + 1
        final_violation.append(entry)
print(zeros)
print(non_zeros)

violation_df = pd.DataFrame(columns=['Case', 'Violation'])
for v_file in final_violation:
    new_row = {'Case': v_file, 'Violation': 1}
    violation_df = pd.concat([violation_df, pd.DataFrame([new_row])], ignore_index=True)
violation_df.to_csv('final_violation.csv', index=False)

'''
