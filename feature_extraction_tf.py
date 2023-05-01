import pandas as pd

class feature_extraction_tf:
    def __init__(self, unique_words_all, count_all, tot_all):
        #self.file = file
        self.unique_words_all = unique_words_all
        self.count_all = count_all
        self.tot_all = tot_all

    def calculate_freq(self, file):
        # For every word in the text
        # if word has not been seen, but in list

        unique_words = []
        count_list = []
        #frequency_list = []

        freq_df = pd.DataFrame(columns=['Words', 'Count', 'T_Freq'])

        list_words = file.split()
        tot_words = len(list_words)  # 1075

        # COUNT TOTAL NUMBER OF WORDS IN ALL DOCUMENTS
        self.tot_all = self.tot_all + tot_words

        # CHECK UNIQUE WORDS IN THE DOC
        for word in list_words:
            if word not in unique_words:
                unique_words.append(word)


        # NUMBER OF OCCURRENCES OF SPECIFIC WORD IN DOC
        for unique in unique_words:
            count = 0
            for word in list_words:
                if unique == word:
                    count = count + 1
            count_list.append(count)


        freq_df['Words'] = unique_words  # 541
        freq_df['Count'] = count_list

        # CHECK UNIQUE WORDS OVERALL (if there are any new unique words in this doc)
        for new_word in unique_words:
            index = unique_words.index(new_word)
            if new_word not in self.unique_words_all:
                #print(new_word)
                self.unique_words_all.append(new_word)
                # add count
                self.count_all.append(count_list[index])
                #print(count_list[index])
            else:
                # get index of position of existing word in unique_words_all
                existing = self.unique_words_all.index(new_word)
                current_count = self.count_all[existing]
                new_count = current_count + count_list[index]
                self.count_all[existing] = new_count

        # CALCULATE FREQUENCY
        freq_df['T_Freq'] = freq_df['Count'] / tot_words
        freq_df = freq_df.sort_values('T_Freq', ascending=False).reset_index(drop=True)

        return freq_df, self.unique_words_all, self.count_all, self.tot_all


    def calculate_all_freq(self):
        all_freqs= pd.DataFrame(columns=['Words', 'Count', 'T_Freq'])
        all_freqs['Words'] = self.unique_words_all
        all_freqs['Count'] = self.count_all

        # CALCULATE FREQUENCY
        all_freqs['T_Freq'] = all_freqs['Count'] / self.tot_all
        all_freqs = all_freqs.sort_values('T_Freq', ascending=False).reset_index(drop=True)
        return all_freqs






