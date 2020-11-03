import matplotlib.pyplot as plt
import pandas as pd
from langdetect import detect_langs
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud


class DataAnalyser:
    def __init__(self):
        # Create stopwords
        self.english_stopwords = stopwords.words('english')
        self.my_stopwords = ["night", "coming", "many", "lots", "seperate", "dogs", "basket", "duvets", "hotel", "iron",
                             "channels", "mamoot", "pillows", "arrival", "use", "coffee", 'square', "red", "bustle",
                             "square", "near", "phone", "toilet", "room", "top", "stay", "really", "Eline", "couldn",
                             "didn", "doesn", "room", "don", "hadn", "hasn", "haven", "isn", "let", "ll", "mustn", "re",
                             "shan", "shouldn", "ve", "wasn", "weren", "won", "wouldn", "go", "aren", "nothing", "like",
                             "would", "back", "told", "need", "even", "one", "especially", "could", "close", "brb"]
        self.stopwords = self.my_stopwords + self.english_stopwords
        self.vect = CountVectorizer(stop_words=self.stopwords)
        self.lr = LogisticRegression()
        self.nb = MultinomialNB()
        self.rf = RandomForestClassifier(criterion='entropy', max_depth=1000, max_features='auto', n_estimators=200,
                                         random_state=42)

    def word_cloud(self, df, label):
        all_reviews = ""
        for row in df.values:
            all_reviews += " " + row[1]
        all_reviews.strip()

        # Generate and show the word cloud
        my_cloud = WordCloud(background_color='white', stopwords=self.stopwords).generate(all_reviews)
        plt.imshow(my_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(label, fontsize=30)
        plt.show()

    def detect_langs(self, df):
        # Detect languages for all reviews
        used_langs = []
        for row in df.values:
            used_langs.append(detect_langs(row[3]))

        return used_langs

    def extract_reviews(self, df):
        # Detect positive and negative reviews
        reviews = []
        for row in df.values:
            reviews.append([row[0], row[3].strip('\n*\\'), 1])
            reviews.append([row[0], row[4].strip('\n*\\'), 0])

        return pd.DataFrame(reviews)

    def lr_score(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['review'], df['rating'], random_state=45, test_size=0.25)

        temp = self.vect.fit_transform(X_train)

        tdif = TfidfTransformer()
        temp2 = tdif.fit_transform(temp)

        model = self.lr.fit(temp2, y_train)

        prediction_data = tdif.transform(self.vect.transform(X_test))
        predicted = model.predict(prediction_data)
        print("LogisticRegression: ", accuracy_score(y_test, predicted))

    def nb_score(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['review'], df['rating'], random_state=45, test_size=0.25)

        X_train_cv = self.vect.fit_transform(X_train)
        X_test_cv = self.vect.transform(X_test)

        tdif = TfidfTransformer()
        temp = tdif.fit_transform(X_test_cv)

        self.nb.fit(X_train_cv, y_train)

        nb_pred = self.nb.predict(temp)
        print('Naive Bayes: ', accuracy_score(y_test, nb_pred))

    def rf_score(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['review'], df['rating'], random_state=67, test_size=0.2)

        X_train_cv = self.vect.fit_transform(X_train)
        X_test_cv = self.vect.transform(X_test)

        tdif = TfidfTransformer()
        temp = tdif.fit_transform(X_test_cv)

        self.rf.fit(X_train_cv, y_train)

        y_pred = self.rf.predict(temp)
        print('Random Forest: ', accuracy_score(y_test, y_pred))

    def rf_gridsearch(self, df):
        X_train, X_test, y_train, y_test = train_test_split(df['review'], df['rating'], random_state=67, test_size=0.2)

        X_train_cv = self.vect.fit_transform(X_train)

        model = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'max_features': ['auto', 'log2'],
            'max_depth': [50, 1000],
            'criterion': ['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train_cv, y_train)

        print(cv_rfc.best_params_)

    def test_sentence(self, sentence):
        test_sentence = [sentence]
        x_sentence = self.vect.transform(test_sentence)
        print(self.rf.predict(x_sentence)[0])
