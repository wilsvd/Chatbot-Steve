import numpy as np
from utility.similarity_matcher import calculate_similarity
from joblib import load

QA_THRESHOLD = 0.7

class QuestionAnswer():

    def __init__(self) -> None:
        self.qa_data = load("../joblibs/qa_dataset.joblib")

    def get_qa_data(self):
        return self.qa_data

    def answer_question(self, query):
        cos = calculate_similarity(self.qa_data, query)
        if cos.max() >= QA_THRESHOLD:
            id_argmax = np.where(cos == np.max(cos, axis=0))
            id = np.random.choice(id_argmax[0])
            return (self.qa_data['text'].loc[id])
        else:
            return ('NOT FOUND')

    def get_top_5_similar(self, query):
        cos = calculate_similarity(self.qa_data, query)
        ind = np.argsort(cos[:, 0])[::-1]   
        res = self.qa_data['text'].loc[ind[0:5]]
        return res
        
