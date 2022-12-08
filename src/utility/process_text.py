
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

SB_STEMMER = SnowballStemmer('english')  # using snowball stemmer
LEMMATISER = WordNetLemmatizer()
STOP_WORDS = stopwords.words('english')

CHANGE_NAME_WORDS = ["swap", "substitute", "switch",
                     "replace", "rename", "change", "call"]

NAME_WORDS = ["like", "called", "call", "me", "change", "changed", "my", "name", "named",
              "please", "rename", "switch", "yes", "sure"]


QUESTION_WORDS = ["what", "which", "who", "where", "why", "when", "how", "am", "is", "are", "was", "were",
                  "being", "been", "be", "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should"]

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class ProcessText():

    def token_stemming(self, tokens):
        new_tokens = []

        for token in tokens:
            new_tokens.append(SB_STEMMER.stem(token))
        return new_tokens

    def token_lemmatisation(self, tokens):
        new_tokens = []

        word_class_map = {
            'NOUN': 'n',
            'VERB': 'v',
            'ADJ': 'a',
            'ADV': 'r'
        }
        # process the lemmatisation with tags
        post = nltk.pos_tag(tokens, tagset='universal')
        for token in post:
            word, tag = token[0], token[1]
            if tag in word_class_map.keys():
                new_tokens.append(LEMMATISER.lemmatize(
                    word, word_class_map[tag]))
            else:
                new_tokens.append(LEMMATISER.lemmatize(word))
        return new_tokens

    def tokenise_text(self, text, type):
        # tokenise
        text_tokens = word_tokenize(text)
        # remove stop words and special signs
        tokens = []
        for word in text_tokens:
            if word not in STOP_WORDS and word.isalpha():
                tokens.append(word)

        if type == 'lemmatisation':
            tokens = self.token_lemmatisation(tokens)
        else:
            tokens == self.token_stemming(tokens)

        return tokens

    def lemmatise_or_stem(self, text, type):
        # tokenise
        text_tokens = word_tokenize(text)
        # remove stop words and special signs
        tokens = []
        for word in text_tokens:
            if word.isalpha():
                tokens.append(word)

        if type == 'lemmatisation':
            tokens = self.token_lemmatisation(tokens)
        else:
            tokens == self.token_stemming(tokens)

        return tokens

    def preprocess_text(self, text="", stopwords=False, type="lemmatisation"):
        text = str(text).lower()
        res_tokens = []
        if stopwords:
            res_tokens = self.tokenise_text(text, type)
        else:
            res_tokens = self.lemmatise_or_stem(text, type)
        return (' ').join(res_tokens)
