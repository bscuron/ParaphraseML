import pandas as pd
import ssl
from nltk import download, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string

DATA_TRAIN_PATH='../data/train_with_label.txt'
DATA_DEV_PATH='../data/dev_with_label.txt'
DATA_TEST_PATH='../data/test_without_label.txt'

def main():
    data_train, data_dev, data_test = get_data()
    print(data_train)
    print(data_dev)
    print(data_test)

# Read and clean the train set, dev set, and test set. Return each in a tuple in the order (train, dev, test)
def get_data():
    delimiter, column_names = '\t+', ['ID', 'SENTENCE_1', 'SENTENCE_2', 'GROUND_TRUTH']
    data_train = clean(pd.read_csv(DATA_TRAIN_PATH, sep=delimiter, engine='python', names=column_names))
    data_dev = clean(pd.read_csv(DATA_DEV_PATH, sep=delimiter, engine='python', names=column_names))
    data_test = clean(pd.read_csv(DATA_TEST_PATH, sep=delimiter, engine='python', names=column_names[:-1]))
    return data_train, data_dev, data_test

# Cleans the text data in columns 'SENTENCE_1' and 'SENTENCE_2'
def clean(df):
    lemmatizer = WordNetLemmatizer()
    ILLEGAL = set(stopwords.words('english') + list(string.punctuation))
    df['SENTENCE_1'] = df['SENTENCE_1'].apply(lambda x: ' '.join([lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in word_tokenize(x) if token not in ILLEGAL and len(token.strip(string.punctuation)) > 1]))
    df['SENTENCE_2'] = df['SENTENCE_2'].apply(lambda x: ' '.join([lemmatizer.lemmatize(token.lower().strip(string.punctuation)) for token in word_tokenize(x) if token not in ILLEGAL and len(token.strip(string.punctuation)) > 1]))
    return df


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
