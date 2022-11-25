import pandas as pd
import numpy as np
from process_text import create_sentence
import json
from joblib import load

class SmallTalk():

    def __init__(self):
        self.classifier = load("./joblibs/intent_classifier.joblib")
        self.tf_idf = load("./joblibs/tfidf_vectorizer.joblib")

    def __get_intent(self, query):
        input_tfidf = self.tf_idf.transform([query])
        result = self.classifier.predict(input_tfidf)
        return result[0]

    def find_response(self, query):
        result = self.__get_intent(query)

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
