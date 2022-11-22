import nltk

from nltk.tokenize import word_tokenize
from process_text import STOP_WORDS

CHANGE_NAME_WORDS = ["swap", "substitute", "switch",
                     "replace", "rename", "change", "call"]

NAME_WORDS = ["like", "called", "call", "me", "change", "changed", "my", "name", "named",
              "please", "rename", "switch", "yes", "sure"]


def is_name_change(input):
    text_tokens = word_tokenize(input)

    for token in set(text_tokens):
        if token in CHANGE_NAME_WORDS:
            return True
    return False


def set_username(input):
    text_tokens = word_tokenize(input)

    username = []
    for token in text_tokens:
        if token not in NAME_WORDS and token.isalpha() and token not in STOP_WORDS:
            username.append(token)

    username = (" ").join(username)

    return username.rstrip()
    # user_name = [i for i in text_tokens if not i.lower() in NAME and i.isalpha() and not i.lower() in stopwords.words('english')]
    # user_name = (' ').join(user_name)
    # return user_name
