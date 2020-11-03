import re

from nltk.stem import PorterStemmer


class DataCleaner:
    def __init__(self):
        self.stemmer = PorterStemmer()
        # nltk.download()

    def stemmed(self, df):
        # Stem every word in the sentence
        df['review'] = df['review'].apply(
            lambda x: " ".join([self.stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())
        return df

    def string_cleaner(self, df, column):
        df[column] = df[column].map(lambda x: x.replace('+', '').replace('\\n', '').replace('*', ''))
        df[column] = df[column].str.strip('\n*\\')
