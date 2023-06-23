# Maastricht University Bachelor Thesis
### Bianca Caissotti di Chiusano (i6245461)
#### NLP-based Feature Extraction and Document Clustering of ECHR Case Violations Regarding Article 10: Freedom of Expression
## Running the code:
To run the whole project and observe how a case law document (.txt) is preprocessed ect. run main.py

To perform clustering or LDA topic modelling: you can run the jupyter notebooks in the models directory (make sure to use the differently processed facts in the csvs **non_violation_csv** and **violation_csv**) 

## About this thesis:
Title: "NLP-based Feature Extraction and Document Clustering of ECHR Case Violations Regarding Article 10: Freedom of Expression"

The present thesis aims at continuing to explore what factors lead to case violations, specifically related to Freedom of Expression, bringing light to new potential conflicts that have not yet been recognized by the Court.

## The files:

#### Folders:
- **non_violation**: folder containing all cases on non-violation of Article 10 (downloaded but not all are valid to use)
- **violation**: folder containing all cases on violation of Article 10 (downloaded but not all are valid to use)
- **non_violation_csv**: Contains:
  - csv with all the non_violation docs that can be used
  - csvs with facts processed in different preprocessing techniques (ready for clustering and LDA topic modelling):
    - non_violation_facts_processed (fully processed)
    - non_violation_facts_lemma (only lemmatization)
    - non_violation_facts_sw (only stop word removal)
    - non_violation_facts_non (no preprocessing)
- **violation_csv**: same as "non_violation_csv" but for violation cases
#### Main: 
  - For every file in DataFrame (Violation or Non-Violation):
    - Text preprocessing
  - Saving CSVs (Prepares documents for clustering and topic modelling)
#### Feature Extraction:
- **Text Preprocessing**:
  - Clean File
  - Extract Facts
  - Pre-processing:
    - Tokenize
    - Normalise
    - Lemmatize
    - English/Legal Stop word removal
- Two files that didn't end up in the final submission but could be used in the future for better control of tf-idf and n_grams compared to using the pre-made models from SKLearn and Gensim:
  - **Frequency_calculator**: Self made TF-IDF to try and get a sense of the unique words and their frequencies. Sklearn TF-IDF is used in the rest of the project.
  - **n_grams and collocations**: n = 1,2,3,4
#### Models:
  These are in the format of jupyter/google collab notebooks as it is easier to run and visualise experiments
- **K-means document clustering**
- **LDA topic modelling**




