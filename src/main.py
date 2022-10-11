import pandas as pd
import numpy as np
import ssl
from nltk import download, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from thefuzz.fuzz import ratio, partial_ratio, token_sort_ratio, token_set_ratio
from difflib import SequenceMatcher

DATA_TRAIN_PATH='../data/train_with_label.txt'
DATA_DEV_PATH='../data/dev_with_label.txt'
DATA_TEST_PATH='../data/test_without_label.txt'
TMP_DIR = '../tmp'
FEATURE_COLUMNS = ['LEVENSHTEIN_DIST', 'COSINE_SIMILARITY', 'LENGTH_DIFFERENCE', 'SHARED_WORDS']

def main():
    print('Reading, cleaning, analyzing data...')
    data_train, data_dev, data_test = get_data()

    # dump dataframes
    data_train.to_csv(f'{TMP_DIR}/data_train_processed.csv')
    data_dev.to_csv(f'{TMP_DIR}/data_dev_processed.csv')
    data_test.to_csv(f'{TMP_DIR}/data_test_processed.csv')

    print(data_train[FEATURE_COLUMNS + ['GROUND_TRUTH']])
    print(data_dev[FEATURE_COLUMNS + ['GROUND_TRUTH']])
    print(data_test[FEATURE_COLUMNS])

    # print("Finding optimal hyperparameters...")
    # param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000, 2000], 'kernel': ['rbf', 'sigmoid']}
    # clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
    # clf = clf.fit(data_train[FEATURE_COLUMNS], data_train['GROUND_TRUTH'])
    # print(clf.best_estimator_)

    clf = make_pipeline(StandardScaler(), SVC(C=10000, class_weight='balanced', kernel='linear'))
    clf.fit(data_train[FEATURE_COLUMNS], data_train['GROUND_TRUTH'])
    y_dev_pred = clf.predict(data_dev[FEATURE_COLUMNS])
    print(accuracy_score(data_dev['GROUND_TRUTH'], y_dev_pred))

# Read and clean the train set, dev set, and test set. Return each in a tuple in the order (train, dev, test)
def get_data():
    delimiter, column_names = '\t+', ['ID', 'SENTENCE_1', 'SENTENCE_2', 'GROUND_TRUTH']
    data_train = extract_features(clean(pd.read_csv(DATA_TRAIN_PATH, sep=delimiter, engine='python', names=column_names)))
    data_dev = extract_features(clean(pd.read_csv(DATA_DEV_PATH, sep=delimiter, engine='python', names=column_names)))
    data_test = extract_features(clean(pd.read_csv(DATA_TEST_PATH, sep=delimiter, engine='python', names=column_names[:-1])))
    return data_train, data_dev, data_test

# Cleans the text data in columns 'SENTENCE_1' and 'SENTENCE_2'
def clean(df):
    lemmatizer = WordNetLemmatizer()
    ILLEGAL = set(stopwords.words('english') + list(string.punctuation))
    df['SENTENCE_1'] = df['SENTENCE_1'].apply(lambda x: ' '.join([lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in word_tokenize(x) if token not in ILLEGAL and len(token.strip(string.punctuation)) > 1]))
    df['SENTENCE_2'] = df['SENTENCE_2'].apply(lambda x: ' '.join([lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in word_tokenize(x) if token not in ILLEGAL and len(token.strip(string.punctuation)) > 1]))
    return df

# TODO
def extract_features(df):
    df['LEVENSHTEIN_DIST'] = get_levenshtein_distance(df)
    df['LENGTH_DIFFERENCE'] = get_length_difference(df)
    df['COSINE_SIMILARITY'] = get_cosine_similarity(df)
    df['SHARED_WORDS'] = get_shared_words(df)
    # df['RATIO'] = get_ratios(df)
    # df['PARTIAL_RATIO'] = get_partial_ratios(df)
    # df['TOKEN_SORT_RATIO'] = get_token_sort_ratios(df)
    # df['TOKEN_SET_RATIO'] = get_token_set_ratios(df)
    # df['RATCLIFF_OBERSHELP'] = get_ratcliff_obershelp(df)
    return df

def get_ratcliff_obershelp(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(SequenceMatcher(None, s1, s2).ratio())
    return ratios

def get_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(ratio(s1, s2))
    return ratios

def get_partial_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(partial_ratio(s1, s2))
    return ratios

def get_token_sort_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(token_sort_ratio(s1, s2))
    return ratios

def get_token_set_ratios(df):
    ratios = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        ratios.append(token_set_ratio(s1, s2))
    return ratios

def get_shared_words(df):
    shared_words_list = []
    for s1, s2, in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        d1 = {}
        for w in s1.split():
            d1[w] = 1 if w not in d1 else d1[w] + 1
        d2 = {}
        for w in s2.split():
            d2[w] = 1 if w not in d2 else d2[w] + 1
        shared_words = set(d1) & set(d2)
        d = {}
        for w in shared_words:
            d[w] = min(d1[w], d2[w])
        shared_words_list.append(sum(d.values()))
    return shared_words_list

def get_cosine_similarity(df):
    vectorizer = TfidfVectorizer()
    corpus = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        corpus.append(s1)
        corpus.append(s2)
    vectorizer.fit(corpus)
    sentence_1_vectors = df['SENTENCE_1'].apply(lambda x: np.array([value for value in vectorizer.transform([x]).A]).flatten())
    sentence_2_vectors = df['SENTENCE_2'].apply(lambda x: np.array([value for value in vectorizer.transform([x]).A]).flatten())
    cosine_simililarity_vector = []
    for vec1, vec2 in zip(sentence_1_vectors, sentence_2_vectors):
        cosine_simililarity_vector.append(cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0])
    return cosine_simililarity_vector

def get_length_difference(df):
    differences = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        differences.append(abs(len(s1) - len(s2)))
    return differences


def get_levenshtein_distance(df):
    distances = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        distances.append(levenshtein_distance(s1, s2))
    return distances

if __name__ == '__main__':
    # Disable SSL check
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # Downlad stop words
    download('stopwords')

    # Run main
    main()
