import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from match_intent import calculate_similarity
from process_text import create_sentence
import json

SM_THRESHOLD = 0.65

# def make_small_talk(sm_data, query):
#     cos = calculate_similarity(sm_data, query)
#     if cos.max() >= SM_THRESHOLD:
#         id_argmax = np.where(cos == np.max(cos, axis=0))
#         id = np.random.choice(id_argmax[0])
#         result = sm_data['answer'].loc[id]
#         return result
#     else:
#         return 'NOT FOUND'

def get_intent(intent_data, query):
    # TF-IDF
    tfidf_vec = TfidfVectorizer(analyzer='word')
    # Document-term matrix
    X_tfidf = tfidf_vec.fit_transform(intent_data['Utterances']).toarray()
    input_tfidf = tfidf_vec.transform([query]).toarray()
    # Dataframe
    df_tfidf = pd.DataFrame(X_tfidf, columns=tfidf_vec.get_feature_names_out())
    # Cosine similarity
    cos = 1 - pairwise_distances(df_tfidf, input_tfidf, metric='cosine')

    if cos.max() >= SM_THRESHOLD:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0])
        intent = intent_data['Intent'].loc[id]
        return intent
    else:
        return 'NOT FOUND'

def find_response(intent_data, query):
    result = get_intent(intent_data, query)

    if result == 'NOT FOUND':
        return 'NOT FOUND'
    
    with open('./responses.json') as f:
        response_data = json.load(f)
        options = response_data[result]
        id = np.random.choice(len(options))
        return options[id]


# After getting intent, I know that it exsits (That is why I was able to get it dumb dumb)
# Now that I have the intent. I need to use it to generate text.
# I should give each intent 2-3 responses that I can randomly select from.
# The question is where am I going to store this data?
# Am I going to have these responses in a dataset? - I feel like I should but then how do I access it.
# If responses is a dataframe then there must be a way to check if the INTENT column contains the intent and if it does then I just
# randomly pick one of the intents.

# My next question though is how do I store multiple responses for a single intent in a dataframe.
# Maybe it is better to have a responses.py file -> I think I will start here.



def replicate_answer(query):
    new_sentence = create_sentence(query)
    return new_sentence
