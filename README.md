# Maastricht University Bachelor Thesis
### Bianca Caissotti di Chiusano (i6245461)
#### NLP-based Feature Extraction and Document Clustering of ECHR Case Violations Regarding Article 10: Freedom of Expression
## Running the code:
To do run the whole project and observe how a case law document (.txt) is preprocessed ect. run main.py

To perform clustering only: run clustering.py (make sure you provide a valid DataFrame containing facts from all the violation/nonviolation documents that you want to cluster)

## About this thesis:
Title: "NLP-based Feature Extraction and Document Clustering of ECHR Case Violations Regarding Article 10: Freedom of Expression"

The present thesis aims at continuing to explore what factors lead to case violations, specifically related to Freedom of Expression, bringing light to new potential conflicts that have not yet been recognized by the Court.

## The files:

#### Folders:
- **non_violation**: folder containing all cases on non-violation of Article 10 (downloaded but not all are valid to use)
- **violation**: folder containing all cases on violation of Article 10 (downloaded but not all are valid to use)
- **non_violation_csv**: Contains:
  - DF - one row per document - contains all processed facts of that document - ready for clustering
  - csv with all the non_violation docs that can be used
  - csv_groups and csv_overall are csvs that were created to check the frequencies of words and self-made groups of similar words (just a start)
- **violation_csv**: same as "non_violation_csv" but for violation cases
#### Main: 
  - For every file in DataFrame (Violation or Non-Violation):
    - Text preprocessing
    - Calculate N-Grams and Frequencies
  - Saving CSVs
  - Prepares documents for clustering
#### Feature Extraction:
- **Frequency_calculator**: Self made TF-IDF to try and get a sense of the unique words and their frequencies. Sklearn TF-IDF is used in the rest of the project.
- **n_grams and collocations**: n = 1,2,3,4
- **Text Preprocessing**:
  - Clean File
  - Extract Facts
  - Pre-processing:
    - Tokenize
    - Normalise
    - Lemmatize
    - English/Legal Stop word removal
  - **Topic Modelling** : TODO
#### Clustering:
- **Word-embedding Techniques**:
  - TF-IDF
  - CountVectorizer
- **K-means clustering**




