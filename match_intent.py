from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from process_text import preprocess_text
import pandas as pd
import numpy as np


def check_intent(name_data, sm_data, qa_data, query):
    name_cos = get_name_similarity(name_data, query)
    sm_cos = get_sm_similarity(sm_data, query)
    qa_cos = get_qa_similarity(qa_data, query)

    return get_most_similar(name_cos, sm_cos, qa_cos)


def get_most_similar(name_cos, sm_cos, qa_cos):
    name_max = name_cos.max()
    sm_max = sm_cos.max()
    qa_max = qa_cos.max()
    print(f"Name: {name_max}, Small Talk: {sm_max}, Q&A: {qa_max} ")

    if (name_max >= sm_max and name_max >= qa_max):
        return ["NAME", name_cos]
    elif (sm_max >= name_max and sm_max >= qa_max):
        return ["SMALLTALK", sm_cos]
    else:
        return ["QUESTION", qa_cos]


def get_name_similarity(name_data, query):
    return calculate_similarity(name_data, query)


def get_sm_similarity(sm_data, query):
    return calculate_similarity(sm_data, query)


def get_qa_similarity(qa_data, query):
    return calculate_similarity(qa_data, query)


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
