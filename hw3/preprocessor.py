
import string
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        # load stopwords
        stop_words = open('/Users/Jennifer/Desktop/hw3/stopwords.txt').read()
        self.tokenize_stop_words = word_tokenize(stop_words.lower())

    # 1. Filter and remove stopwords
    # 2. Apply tokenization
    def stopwords_filter(self, text_file):
        # Read text from file
        text = open(text_file).read()
        tokenize_words = word_tokenize(text)

        tokenize_words_without_stopwords = []
        for word in tokenize_words:
            lowercase_word = word.lower()
            if lowercase_word not in self.tokenize_stop_words and not all(char in string.punctuation for char in lowercase_word):
                tokenize_words_without_stopwords.append(word)
        return tokenize_words_without_stopwords

    # 2. Apply stemming and lemmatization
    def lemmatize_stem(self, tokenize_words_without_stopwords):
        lemmatizer = WordNetLemmatizer()
        # stemmer = PorterStemmer()
        lemma_list = []
        for word in tokenize_words_without_stopwords:
            lemma = lemmatizer.lemmatize(word, "v")
            # stem = stemmer.stem(word)
            lemma_list.append(lemma)
        return lemma_list

    # 3. Apply NER
    def ner_process(self, lemma_list):
        nlp = spacy.load('en_core_web_sm')
        text = ' '.join(lemma_list)
        doc = nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]

        # Get the NER list (no duplicate)
        ner_list = []
        for entity, label in entities:
            if entity in ner_list:
                pass
            else:
                ner_list.append(entity)
        return ner_list

        # for entity, label in entities:
        #     print(f"Entity: {entity}, Label: {label}")

    # 4. Use a sliding window approach to merge remaining phrases that belong together.
    # Apply n-grams
    def ngram_process(self, lemma_list, n):
        ngrams = nltk.ngrams(lemma_list, n)

        # Create a dictionary to store the frequency counts
        frequency_counts = {}
        # Calculate the frequency of each n-gram
        for gram in ngrams:
            if gram in frequency_counts:
                frequency_counts[gram] += 1
            else:
                frequency_counts[gram] = 1

        # Get the ngram list (no duplicate)
        min_treshold = 2
        ngram_list = []
        final_ngram_list = []
        for gram, frequency in frequency_counts.items():
            if frequency >= min_treshold:
                ngram_list.append(gram)
            # print(f"{gram}: {frequency}")
        final_ngram_list = [' '.join(ng) for ng in ngram_list]
        return final_ngram_list

    # Merge NER and n-gram results to word bags
    def ner_ngram_merge(self, ner_list, two_gram, three_gram, four_gram):

        merge_list = ner_list + two_gram + three_gram + four_gram
        word_bags = list(set(merge_list))
        return word_bags

    # Combine the functions above. Users can use this function to process all the text cleaning
    def get_word_bags(self, load_files):
        data_preprocess = Preprocessor()
        stop_words = data_preprocess.stopwords_filter(load_files)
        lemma_list = data_preprocess.lemmatize_stem(stop_words)
        ner_list = data_preprocess.ner_process(lemma_list)
        two_gram = data_preprocess.ngram_process(lemma_list, 2)
        three_gram = data_preprocess.ngram_process(lemma_list, 3)
        four_gram = data_preprocess.ngram_process(lemma_list, 4)
        word_bags = data_preprocess.ner_ngram_merge(ner_list, two_gram, three_gram, four_gram)

        return word_bags






