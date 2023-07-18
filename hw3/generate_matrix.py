
import pandas as pd
import math


class MatrixGenerator:
    def __init__(self):
        pass

    def generate_document_term_matrix(self, df_list, txt_files):
        merged_df = pd.concat(df_list)
        merged_df.fillna(0, inplace=True)
        merged_df.set_index(pd.Index(txt_files), inplace=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        return merged_df

    def compute_tf(self, word_dict, no_stop_words):
        tf_dict = {}
        no_stop_words_count = len(no_stop_words)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(no_stop_words_count)
        return tf_dict

    def compute_idf(self, word_dict_list):
        idf_dict = {}
        n = len(word_dict_list) # count the number of files

        # count the number of documents that contain a word w
        for word_dict in word_dict_list:
            for word, count in word_dict.items():
                if count > 0:
                    idf_dict.setdefault(word, 0)
                    idf_dict[word] += 1

        # divide n by denominator above, take the log of that
        for word, count in idf_dict.items():
            idf_dict[word] = math.log(n/float(count))

        return idf_dict

    def compute_tfidf(self, tf_dict, idf_dict):
        tfidf_dict = {}
        for word, count in tf_dict.items():
            tfidf_dict[word] = round(count * idf_dict[word], 4)
        return tfidf_dict

    def generate_tfidf_matrix(self, tfidf_dict_list, txt_files):
        tfidf_df = pd.DataFrame(tfidf_dict_list)
        tfidf_df.fillna(0, inplace=True)
        tfidf_df.set_index(pd.Index(txt_files), inplace=True)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        return tfidf_df

    def generate_keywords(self, sub_lists, sub_num):
        key_words_count = {}
        for tfidf_dict in sub_lists[sub_num]:
            for word, freq in tfidf_dict.items():
                if freq > 0.02:
                    key_words_count.setdefault(word, 0)
                    key_words_count[word] += 1

        # sorted_key_words_count = dict(sorted(key_words_count.items(), key=lambda x: x[1], reverse=True))
        sorted_keys = sorted(key_words_count, key=key_words_count.get, reverse=True)[:10]
        return sorted_keys







