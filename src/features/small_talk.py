import pandas as pd
import numpy as np
import json
from joblib import load

class SmallTalk():

    def __init__(self):
        self.classifier = load("../joblibs/intent_classifier.joblib")
        self.tf_idf = load("../joblibs/tfidf_vectorizer.joblib")

    def __get_intent(self, query):
        input_tfidf = self.tf_idf.transform([query])
        result = self.classifier.predict(input_tfidf)
        return result[0]

    def find_response(self, query):
        result = self.__get_intent(query)
        
        with open('../datasets/sm_responses.json') as f:
            response_data = json.load(f)
            options = response_data[result]
            id = np.random.choice(len(options))
            return options[id]