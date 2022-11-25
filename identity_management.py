from nltk.tokenize import word_tokenize
from process_text import STOP_WORDS
from match_intent import calculate_similarity
from joblib import load

CHANGE_NAME_WORDS = ["swap", "substitute", "switch",
                     "replace", "rename", "change", "call"]

NAME_WORDS = ["like", "called", "call", "me", "change", "changed", "my", "name", "named",
              "please", "rename", "switch", "yes", "sure"]

NAME_THRESHOLD = 0.8

class IdentityManagement():

    def __init__(self) -> None:
        self.name_data = load("./joblibs/name_dataset.joblib")

    def is_name_change(self, input):
        text_tokens = word_tokenize(input)

        for token in set(text_tokens):
            if token in CHANGE_NAME_WORDS:
                return True
        return False


    def set_username(self, input):
        text_tokens = word_tokenize(input)

        username = []
        for token in text_tokens:
            if token not in NAME_WORDS and token.isalpha() and token not in STOP_WORDS:
                username.append(token)

        username = (" ").join(username)

        return username.rstrip()


    def get_name_similarity(self, query):
        cos = calculate_similarity(self.name_data, query)
        if cos.max() >= NAME_THRESHOLD:
            return "NAME"
        else:
            return "NOT FOUND"
