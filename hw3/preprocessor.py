
import string
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


class Preprocessor:
    def __init__(self):
        # load stopwords
        stop_words = open('stopwords.txt').read()
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

        updated_lemma_list = []
        for word in lemma_list:
            if "'s" not in word:
                updated_lemma_list.append(word)
        lemma_list = updated_lemma_list
        return lemma_list

    # 3. Apply NER
    def ner_process(self, lemma_list):
        nlp = spacy.load('en_core_web_sm')
        text = ' '.join(lemma_list)
        doc = nlp(text)
        entities = [(entity.text, entity.label_) for entity in doc.ents]

        ner_list = []
        for text, label in entities:
            ner_list.append(text.lower())
        return ner_list

        # for entity, label in entities:
        #     print(f"Entity: {entity}, Label: {label}")

    # 4. Use a sliding window approach to merge remaining phrases that belong together.
    # Apply n-grams
    def ngram_process(self, lemma_list, n, threshold):
        lower_lemma_list = []
        for lemma in lemma_list:
            lower_lemma_list.append(lemma.lower())

        ngrams = nltk.ngrams(lower_lemma_list, n)

        # Create a dictionary to store the frequency counts
        ngram_freq_counts = {}
        # Calculate the frequency of each n-gram
        for gram in ngrams:
            if gram in ngram_freq_counts:
                ngram_freq_counts[gram] += 1
            else:
                ngram_freq_counts[gram] = 1

        # Get the ngram list
        min_treshold = threshold
        new_ngram_freq_counts = {}
        for gram, frequency in ngram_freq_counts.items():
            if frequency >= min_treshold:
                new_ngram_freq_counts[gram] = frequency
            # print(f"{gram}: {frequency}")

        # Turn the key into string instead of tuples
        final_ngram_freq_counts = {}
        for key, value in new_ngram_freq_counts.items():
            new_key = ' '.join(key)
            final_ngram_freq_counts[new_key.lower()] = value

        ngram_list = []
        for gram, frequency in final_ngram_freq_counts.items():
            ngram_list.extend([gram] * frequency)
        return ngram_list

    # Merge NER and n-gram results to word bags
    def ner_ngram_merge(self, one_gram, two_gram, three_gram, ner_list):
        word_set = set(one_gram).union(set(two_gram)).union(set(three_gram)).union(set(ner_list))
        word_dict = dict.fromkeys(word_set, 0)

        updated_ner_list = []
        for word in ner_list:
            if word not in one_gram and word not in two_gram and word not in three_gram:
                updated_ner_list.append(word)

        for word in one_gram:
            word_dict[word] += 1
        for word in two_gram:
            word_dict[word] += 1
        for word in three_gram:
            word_dict[word] += 1
        for word in updated_ner_list:
            word_dict[word] += 1
        return word_dict

    # This function merges the functions above
    def get_word_bags(self, load_files):
        data_preprocess = Preprocessor()
        no_stop_words = data_preprocess.stopwords_filter(load_files)
        lemma_list = data_preprocess.lemmatize_stem(no_stop_words)
        ner_list = data_preprocess.ner_process(lemma_list)
        one_gram = data_preprocess.ngram_process(lemma_list, 1, 3)
        two_gram = data_preprocess.ngram_process(lemma_list, 2, 2)
        three_gram = data_preprocess.ngram_process(lemma_list, 3, 2)
        word_dict = data_preprocess.ner_ngram_merge(one_gram, two_gram, three_gram, ner_list)
        return word_dict






