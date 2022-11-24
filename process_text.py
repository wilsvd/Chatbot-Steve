
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

SB_STEMMER = SnowballStemmer('english')  # using snowball stemmer
LEMMATISER = WordNetLemmatizer()
STOP_WORDS = stopwords.words('english')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


def token_stemming(tokens):
    new_tokens = []

    for token in tokens:
        new_tokens.append(SB_STEMMER.stem(token))
    return new_tokens


def token_lemmatisation(tokens):
    new_tokens = []

    posmap = {
        'NOUN': 'n',
        'VERB': 'v',
        'ADJ': 'a',
        'ADV': 'r'
    }
    # process the lemmatisation with tags
    post = nltk.pos_tag(tokens, tagset='universal')
    for token in post:
        word, tag = token[0], token[1]
        if tag in posmap.keys():
            new_tokens.append(LEMMATISER.lemmatize(word, posmap[tag]))
        else:
            new_tokens.append(LEMMATISER.lemmatize(word))
    return new_tokens


def tokenise_text(text, type):
    # tokenise
    text_tokens = word_tokenize(text)
    # remove stop words and special signs
    tokens = []
    for word in text_tokens:
        if word not in STOP_WORDS and word.isalpha():
            tokens.append(word)

    if type == 'lemmatisation':
        tokens = token_lemmatisation(tokens)
    else:
        tokens == token_stemming(tokens)

    return tokens

def lemmatise_or_stem(text, type):
    # tokenise
    text_tokens = word_tokenize(text)
    # remove stop words and special signs
    tokens = []
    for word in text_tokens:
        if word.isalpha():
            tokens.append(word)

    if type == 'lemmatisation':
        tokens = token_lemmatisation(tokens)
    else:
        tokens == token_stemming(tokens)

    return tokens


def preprocess_text(text = "", stopwords = False, type = "lemmatisation"):
    text = str(text).lower()
    res_tokens = []
    if stopwords:
        res_tokens = tokenise_text(text, type)
    else:
        res_tokens = lemmatise_or_stem(text, type)
    return (' ').join(res_tokens)


QUESTION_WORDS = ["what", "which", "who", "where", "why", "when", "how", "am", "is", "are", "was", "were",
                  "being", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should"]


def create_sentence(query):

    text_tokens = word_tokenize(query)
    # process the lemmatisation with tags
    post = nltk.pos_tag(text_tokens, tagset='universal')
    new_string = []

    for key, token in enumerate(post):
        word = token[0].lower()
        if key == 0 and token[0].lower() in QUESTION_WORDS:
            continue

        if word == "you":
            word = "I"
        elif word == "I":
            word = "You"
        new_string.append(word)

    result = " ".join(new_string)
    return result
