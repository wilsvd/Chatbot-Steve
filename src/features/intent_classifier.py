from joblib import load, dump
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utility.process_text import ProcessText


class TrainClassifier():

    def __init__(self) -> None:
        self.intent_data = load("../joblibs/intent_dataset.joblib")
        self.process_text = ProcessText()

    def dataset_characteristics(self):
        print(self.intent_data.isna().sum())
        print(self.intent_data.Intent.nunique())
        print(self.intent_data.head())

    def clean_dataset(self):
        self.intent_data['Utterances'] = self.intent_data['Utterances'].apply(
            self.process_text.preprocess_text, stopwords=False, type='lemmatisation')

    def split_dataset(self):
        self.dataDoc = self.intent_data['Utterances'].values.tolist()
        dataClass = self.intent_data['Intent'].values.tolist()

        self.set_dataClass(dataClass)

        # Before digitizing the data using the Tfidf scoring method, we separate the data as training and testing.
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.dataDoc, self.dataClass, test_size=0.33, random_state=69, stratify=dataClass)

    def vectorise_data(self):
        # min_df: It is used to ignore terms that rarely appear. Currently if a term occurs in less than 2 documents, it will be ignored.
        tfidf_vectorizer = TfidfVectorizer(analyzer='word')
        self.x_train_tfidf = tfidf_vectorizer.fit_transform(self.x_train)
        self.x_test_tfidf = tfidf_vectorizer.transform(self.x_test)
        dump(tfidf_vectorizer, "../joblibs/tfidf_vectorizer.joblib")

    def train_svm(self):
        svc_clf = SVC(kernel='linear', random_state=69).fit(
            self.x_train_tfidf, self.y_train)
        dump(svc_clf, "../joblibs/intent_classifier.joblib")

    def get_dataClass(self):
        return self.dataClass

    def set_dataClass(self, dataClass):
        self.dataClass = dataClass
