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

DATA_TRAIN_PATH='../data/train_with_label.txt'
DATA_DEV_PATH='../data/dev_with_label.txt'
DATA_TEST_PATH='../data/test_without_label.txt'
FEATURE_COLUMNS = ['LEVENSHTEIN_DIST', 'COSINE_SIMILARITY', 'LENGTH_DIFFERENCES']

def main():
    print('Reading, cleaning, analyzing data...')
    data_train, data_dev, data_test = get_data()
    # print(data_train[['LEVENSHTEIN_DIST', 'COSINE_SIMILARITY', 'GROUND_TRUTH']])
    # print(data_dev[['LEVENSHTEIN_DIST', 'COSINE_SIMILARITY', 'GROUND_TRUTH']])
    # print(data_test[['LEVENSHTEIN_DIST', 'COSINE_SIMILARITY', 'GROUND_TRUTH']])

    # print("Finding optimal hyperparameters...")
    # param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000, 2000], 'kernel': ['rbf', 'sigmoid']}
    # clf = GridSearchCV(SVC(class_weight='balanced'), param_grid)
    # clf = clf.fit(data_train[FEATURE_COLUMNS], data_train['GROUND_TRUTH'])
    # print(clf.best_estimator_)


    clf = SVC(C=10000, class_weight='balanced')
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
    df['LENGTH_DIFFERENCES'] = get_length_difference(df)
    df['COSINE_SIMILARITY'] = get_cosine_similarity(df)
    return df

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
