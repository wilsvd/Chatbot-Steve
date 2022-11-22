from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from process_text import preprocess_text
import pandas as pd
import numpy as np

def calculate_similarity(dataset, query):
    # TF-IDF
    tfidf_vec = TfidfVectorizer(analyzer='word')
    # Document-term matrix
    X_tfidf = tfidf_vec.fit_transform(dataset['question']).toarray()
    input_tfidf = tfidf_vec.transform([query]).toarray()
    # Dataframe
    df_tfidf = pd.DataFrame(X_tfidf, columns=tfidf_vec.get_feature_names_out())
    # Cosine similarity
    cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric='cosine')
    return cos
