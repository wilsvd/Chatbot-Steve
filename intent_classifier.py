# %%
from joblib import load, dump
import pandas as pd
import numpy as np
# Loads my Intent Dataframe
intent_data = load("./joblibs/intent_dataset.joblib")
# %%
intent_data.isna().sum()
# %%
intent_data.Intent.nunique()
# %%
intent_data.head()
# %%
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+') 
punct_re=lambda x :" ".join(tokenizer.tokenize(x.lower())) 
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

intent_data['Utterances'] = intent_data['Utterances'].apply(lemmatize_text) 
intent_data['Utterances'] = intent_data['Utterances'].apply(lambda x : " ".join(x)) 

intent_data['Utterances'] = intent_data['Utterances'].str.replace(r'\S*@\S*\s?', '', regex=True) 
intent_data['Utterances'] = intent_data['Utterances'].str.replace(r'[^\w\s]', '', regex=True) 
intent_data
# %%

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score 
dataDoc = intent_data['Utterances'].values.tolist() 
dataClass = intent_data['Intent'].values.tolist()

# Before digitizing the data using the Tfidf scoring method, we separate the data as training and testing.
x_train, x_test, y_train, y_test = train_test_split(dataDoc, dataClass, test_size = 0.33, random_state = 69, stratify=dataClass)

#tfidf işlemi
tfidf_vectorizer = TfidfVectorizer(analyzer='word') # min_df: It is used to ignore terms that rarely appear. Currently if a term occurs in less than 2 documents, it will be ignored.
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)
dump(tfidf_vectorizer, "./joblibs/tfidf_vectorizer.joblib")

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker

# %%
from sklearn.svm import SVC
svc_clf = SVC(kernel = 'linear').fit(x_train_tfidf,y_train)
pred_train_svc = svc_clf.predict(x_train_tfidf)
pred_test_svc = svc_clf.predict(x_test_tfidf)
# %%
print(np.shape(x_test_tfidf))
print(np.shape(x_train_tfidf))
print(np.shape(pred_test_svc))
print(np.shape(pred_train_svc))
# %%
# The training score and test score are compared to see if our model has overfitted. If the values are close, there is no overfitting.
# Training score
print('Support Vector Machine training dataset accuracy: {0:0.4f}'. format(metrics.accuracy_score(y_train, pred_train_svc)))

# Test score
print('Support Vector Machine test dataset accuracy: {0:0.4f}'.format(metrics.accuracy_score(y_test, pred_test_svc)))

# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Support vector training accuracy
train_con_matrix = (confusion_matrix(y_train,pred_train_svc))
train_class_report = (classification_report(y_train,pred_train_svc))
train_acc_score = (accuracy_score(y_train, pred_train_svc))
# Support vector testing accuracy
test_con_matrix = (confusion_matrix(y_test,pred_test_svc))
test_class_report = (classification_report(y_test,pred_test_svc))
test_acc_score = (accuracy_score(y_test, pred_test_svc))

# %%
print(train_con_matrix)
print(train_class_report)
print(train_acc_score)
# %%
print(test_con_matrix)
print(test_class_report)
print(test_acc_score)
# %%
dump(svc_clf, "./joblibs/intent_classifier.joblib")
# %%

