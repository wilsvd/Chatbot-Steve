from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from features.intent_classifier import TrainClassifier

from joblib import load

class TestClassifier():
    def __init__(self) -> None:

        self.train_classifier = TrainClassifier()
        self.train_classifier.dataset_characteristics()
        self.train_classifier.clean_dataset()
        self.train_classifier.split_dataset()
        self.train_classifier.vectorise_data()
        self.train_classifier.train_svm()
        self.classifier = load("../joblibs/intent_classifier.joblib")
        self.tf_idf = load("../joblibs/tfidf_vectorizer.joblib")
    
    def cross_validation(self):
        tfidf_vectorizer = TfidfVectorizer(analyzer='word') # min_df: It is used to ignore terms that rarely appear. Currently if a term occurs in less than 2 documents, it will be ignored.
        vec_dataDoc = tfidf_vectorizer.fit_transform(self.train_classifier.dataDoc)
        dataClass = self.train_classifier.get_dataClass()

        clf = SVC(kernel='linear', random_state=69)
        scores = cross_val_score(clf, vec_dataDoc, dataClass, cv=5, scoring='f1_macro')
        print("Cross Validation: %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    def model_accuracy(self):
        clf = self.classifier
        x_train_tfidf = self.train_classifier.x_train_tfidf
        x_test_tfidf = self.train_classifier.x_test_tfidf

        y_train = self.train_classifier.y_train
        y_test = self.train_classifier.y_test

        pred_train_svc = clf.predict(x_train_tfidf)
        pred_test_svc = clf.predict(x_test_tfidf)

        print('Support Vector Machine training dataset accuracy: {0:0.4f}'. format(accuracy_score(y_train, pred_train_svc)))
        print('Support Vector Machine test dataset accuracy: {0:0.4f}'.format(accuracy_score(y_test, pred_test_svc)))


test_classifier = TestClassifier()
test_classifier.cross_validation()
test_classifier.model_accuracy()

# # %%
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# # Support vector training accuracy
# train_con_matrix = (confusion_matrix(y_train,pred_train_svc))
# train_class_report = (classification_report(y_train,pred_train_svc))
# train_acc_score = (accuracy_score(y_train, pred_train_svc))
# # Support vector testing accuracy
# test_con_matrix = (confusion_matrix(y_test,pred_test_svc))
# test_class_report = (classification_report(y_test,pred_test_svc))
# test_acc_score = (accuracy_score(y_test, pred_test_svc))

# # %%
# print(train_con_matrix)
# print(train_class_report)
# print(train_acc_score)
# # %%
# print(test_con_matrix)
# print(test_class_report)
# print(test_acc_score)
# # %%
# # %%

