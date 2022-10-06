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

# TODO: extract the features of the dataset (things that will help with prediction. Such as: common words, differences in length of sentences, etc.)
# Creates new columns in the given dataframe that will help with predictions
def extract_features(df):
    df['COMMON_WORD_COUNT'] = get_common_word_counts(df)
    df['LENGTH_DIFFERENCE'] = get_length_differences(df)
    df['WORD_DIFFERENCE'] = get_word_differences(df)
    return df

# Returns the number of words different between both sentences (for each sample)
def get_word_differences(df):
    word_differences = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        word_differences.append(abs(len(s1.split()) - len(s2.split())))
    return word_differences

# Returns the difference in length between both sentences (for each sample)
def get_length_differences(df):
    length_differences = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        length_differences.append(abs(len(s1) - len(s2)))
    return length_differences

# Returns the number of common words between both sentences (for each sample)
# TODO: look into making this a 'score' by dividing by the longest sentence
# length. This would take into account sentences that are short and long
# equally.
def get_common_word_counts(df):
    common_counts = []
    for s1, s2 in zip(df['SENTENCE_1'], df['SENTENCE_2']):
        d1, d2 = {}, {}
        for w in s1.split():
            d1[w] = 1 if w not in d1 else d1[w] + 1
        for w in s2.split():
            d2[w] = 1 if w not in d2 else d2[w] + 1
        common_set = set(d1.keys()) & set(d2.keys())
        count = 0
        for element in common_set:
            count += min(d1[element], d2[element])
        common_counts.append(count)
    return common_counts

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
