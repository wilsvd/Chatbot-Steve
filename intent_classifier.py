# %%
from joblib import load, dump
import pandas as pd
import numpy as np
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
x_train, x_test, y_train, y_test = train_test_split(dataDoc, dataClass, test_size = 0.2, random_state = 1, stratify=dataClass)

#tfidf işlemi
tfidf_vectorizer = TfidfVectorizer(analyzer='word') # min_df: It is used to ignore terms that rarely appear. Currently if a term occurs in less than 2 documents, it will be ignored.
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)
dump(tfidf_vectorizer, "./joblibs/tfidf_vectorizer.joblib")

# %%
from sklearn.linear_model import LogisticRegression # Imported to use logistic regression.
lr = LogisticRegression(solver='saga', random_state=42,multi_class='multinomial', max_iter=1000)
lr_clf = lr.fit(x_train_tfidf, y_train) 
pred_test_lr = lr_clf.predict(x_test_tfidf)

# %%
type(lr_clf)
# %%
# The training score and test score are compared to see if our model has overfitted. If the values are close, there is no overfitting.
# Training score
pred_train_lr = lr_clf.predict(x_train_tfidf) 
# %%
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Logistic Regression Model Test Dataset Accuracy
confusion_matrix(y_test,pred_test_lr)
print(classification_report(y_test,pred_test_lr))
print(accuracy_score(y_test, pred_test_lr))
# %%
dump(lr_clf, "./joblibs/intent_classifier.joblib")
# %%

