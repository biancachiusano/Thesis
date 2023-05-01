import pandas as pd

class feature_extraction_tf:
    def __init__(self, file):
        self.file = file

    def calculate_freq(self):
        # For every word in the text
        # if word has not been seen, but in list

        unique_words = []
        count_list = []
        frequency_list = []
        # Document frequency?
        freq_df = pd.DataFrame(columns=['Words', 'Count', 'T_Freq'])

        list_words = self.file.split()
        tot_words = len(list_words)  # 1075

        # CHECK UNIQUE WORDS IN THE DOC
        for word in list_words:
            if word not in unique_words:
                unique_words.append(word)

        # NUMBER OF OCCURENCES OF SPECIFIC WORD IN DOC
        for unique in unique_words:
            count = 0
            for word in list_words:
                if unique == word:
                    count = count + 1
            count_list.append(count)

        freq_df['Words'] = unique_words  # 541
        freq_df['Count'] = count_list

        # CALCULATE FREQUENCY
        freq_df['T_Freq'] = freq_df['Count'] / tot_words
        freq_df = freq_df.sort_values('T_Freq', ascending=False).reset_index(drop=True)
        return freq_df