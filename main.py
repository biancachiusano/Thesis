from clustering.clustering import clustering
from feature_extraction.frequency_calculator import frequency_calculator
from feature_extraction.text_preprocessing import text_preprocessing
import pandas as pd

from feature_extraction.topic_modeling import topic_modeling

final_non_violation = pd.read_csv('non_violation_csv/final_non_violation.csv')
final_non_violation = final_non_violation.reset_index()

final_violation = pd.read_csv('violation_csv/final_violation.csv')
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
            'mr', 'lawyer', 'supreme', 'judicial', 'ha', 'wa', 'european','union','person','right','freedom','expression',
            'see','paragraph','read','follows','article','human','section','criminal','police','abdullah','öcalan','turkish',
            'turkey','istanbul','public','prosecutor','russian federation','public','event','administrative','offence',
            'moscow','proceeding','decision']

month_sw = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november'
            , 'december']

frequencies = frequency_calculator(unique_words_all, count_all, tot_all)
clusters = clustering()

#trial = ["001-211777.txt", "001-219042.txt", "001-201896"]
#trial = ["001-201896.txt"]
for filename in final_non_violation['Case']:
#for filename in final_violation['Case']:
#for filename in trial:
    trial = text_preprocessing(filename)
    cleaned = trial.clean()
    #print("CLEANED: " + cleaned)
    facts = trial.get_facts(cleaned)
    #print("FACTS: " + facts)
    # Facts can be processed differently (For experiments)
    # s_w = True/False --> Remove/No Removal of stop words
    # lemma = True/False --> Apply/No Lemmatization
    processed = trial.preprocess_text(facts, legal_sw, month_sw, s_w=False, lemma=False)
    #print("PROCESSED: " + processed)


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

# CSVs for experiments
#viol_overall_df = frequencies.calculate_all_freq()
#viol_overall_df.to_csv('violation_overall.csv', index=True)

#viol_groups_df = frequencies.create_groups(viol_overall_df)
#viol_groups_df.to_csv('violation_groups.csv')

# Preparing Facts for Topic Modeling and Document Clustering (For experiments)
#violation_facts = clusters.k_means(all_text=all_processed_facts)
#violation_facts.to_csv('violation_csv/violation_facts_processed.csv')
#violation_facts.to_csv('violation_csv/violation_facts_lemma.csv')
#violation_facts.to_csv('violation_csv/violation_facts_sw.csv')
#violation_facts.to_csv('violation_csv/violation_facts_none.csv')

non_violation_facts = clusters.k_means(all_text=all_processed_facts)
#non_violation_facts.to_csv('non_violation_csv/non_violation_facts_processed.csv')
#non_violation_facts.to_csv('non_violation_csv/non_violation_facts_lemma.csv')
#non_violation_facts.to_csv('non_violation_csv/non_violation_facts_sw.csv')
non_violation_facts.to_csv('non_violation_csv/non_violation_facts_none.csv')

# Topic Modelling
#violation_tm = pd.read_csv('violation_csv/violation_facts_processed.csv')
#facts = violation_tm['Facts'].tolist()
#tm = topic_modeling(facts)
#facts_words = tm.organise()
#print(len(facts_words))

#data_bigram, bigram_phases, bigram = tm.bi_grams(facts_words)
#data_bigram_trigram = tm.tri_grams(facts_words, data_bigram, bigram_phases, bigram)
#corpus, id2word = tm.tf_idf(data_bigram_trigram)
#tm.perform_lda(corpus, id2word)

# This piece of code was to filter the documents (downloaded from scraping) that actually contained text and were not empty
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
