import os
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

BASE_DIR = os.getcwd()

special_stop_words = []
stop_words_path = BASE_DIR + "/stop_Word.txt"
with open(stop_words_path, "rb") as f:
   for line in f.readlines():
      # print(f"type l is {type(l)}")
      special_stop_words.append(str(line.decode("utf-8")).removesuffix("\n"))


nltk_stop_words = set(stopwords.words('english'))

# Define the preprocessing steps
def preprocess(document,stop_words=special_stop_words):
    # Tokenization
    tokens = nltk.word_tokenize(document)

    # Remove stopwords
    # stop_words = special_stop_words
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Remove punctuation and special characters
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]

    # Remove empty tokens
    tokens = [token for token in tokens if token]

    original_string = " ".join(tokens)

    return original_string

def apply_preprocess_on_dataset(dataset):
    # Preprocess the text data
    preprocessed_data = []
    for text in dataset:
        preprocessed_data.append(preprocess(text))

    return preprocessed_data