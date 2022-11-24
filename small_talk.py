import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from match_intent import calculate_similarity
from process_text import create_sentence
import json

SM_THRESHOLD = 0.5

def make_small_talk(sm_data, query):
    cos = calculate_similarity(sm_data, query)

    if cos.max() >= SM_THRESHOLD:
        id_argmax = np.where(cos == np.max(cos, axis=0))
        id = np.random.choice(id_argmax[0])
        result = sm_data['answer'].loc[id]
        return result
    else:
        return 'NOT FOUND'

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




# if (intent == "smalltalk_agent_acquaintance"):
#         pass
#     elif (intent == "smalltalk_agent_age"):
#         pass
#     elif (intent == "smalltalk_agent_annoying"):
#         pass
#     elif (intent == "smalltalk_agent_answer_my_question"):
#         pass
#     elif (intent == "smalltalk_agent_bad"):
#         pass
#     elif (intent == "smalltalk_agent_be_clever"):
#         pass
#     elif (intent == "smalltalk_agent_beautiful"):
#         pass
#     elif (intent == "smalltalk_agent_birth_date"):
#         pass
#     elif (intent == "smalltalk_agent_boring"):
#         pass
#     elif (intent == "smalltalk_agent_boss"):
#         pass
#     elif (intent == "smalltalk_agent_busy"):
#         pass
#     elif (intent == "smalltalk_agent_chatbot"):
#         pass
#     elif (intent == "smalltalk_agent_clever"):
#         pass
#     elif (intent == "smalltalk_agent_crazy"):
#         pass
#     elif (intent == "smalltalk_agent_fired"):
#         pass
#     elif (intent == "smalltalk_agent_funny"):
#         pass
#     elif (intent == "smalltalk_agent_good"):
#         pass
#     elif (intent == "smalltalk_agent_happy"):
#         pass
#     elif (intent == "smalltalk_agent_hungry"):
#         pass
#     elif (intent == "smalltalk_agent_marry_user"):
#         pass
#     elif (intent == "smalltalk_agent_my_friend"):
#         pass
#     elif (intent == "smalltalk_agent_occupation"):
#         pass
#     elif (intent == "smalltalk_agent_origin"):
#         pass
#     elif (intent == "smalltalk_agent_ready"):
#         pass
#     elif (intent == "smalltalk_agent_real"):
#         pass
#     elif (intent == "smalltalk_agent_right"):
#         pass
#     elif (intent == "smalltalk_confirmation_yes"):
#         pass
#     elif (intent == "smalltalk_agent_sure"):
#         pass
#     elif (intent == "smalltalk_agent_talk_to_me"):
#         pass
#     elif (intent == "smalltalk_agent_there"):
#         pass
#     elif (intent == "smalltalk_appraisal_bad"):
#         pass
#     elif (intent == "smalltalk_appraisal_good"):
#         pass
#     elif (intent == "smalltalk_appraisal_no_problem"):
#         pass
#     elif (intent == "smalltalk_appraisal_thank_you"):
#         pass
#     elif (intent == "smalltalk_appraisal_welcome"):
#         pass
#     elif (intent == "smalltalk_appraisal_well_done"):
#         pass
#     elif (intent == "smalltalk_confirmation_cancel"):
#         pass
#     elif (intent == "smalltalk_confirmation_no"):
#         pass
#     elif (intent == "smalltalk_dialog_hold_on"):
#         pass
#     elif (intent == "smalltalk_dialog_hug"):
#         pass
#     elif (intent == "smalltalk_dialog_i_do_not_care"):
#         pass
#     elif (intent == "smalltalk_dialog_sorry"):
#         pass
#     elif (intent == "smalltalk_dialog_what_do_you_mean"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass
#     elif (intent == "smalltalk_agent_REMOVE_THIS_STUFF_RN_FOR_EVERYONE_SAKE"):
#         pass

# ['smalltalk_dialog_wrong', ]
# ""
# ""
# ""
# ""
# ""
# ""
# ""
# ""
# ""
# ""
# "smalltalk_emotions_ha_ha"
# "smalltalk_emotions_wow"
# "smalltalk_greetings_bye"
# "smalltalk_greetings_goodevening"
# "smalltalk_greetings_goodmorning"
# "smalltalk_greetings_goodnight"
# "smalltalk_greetings_hello"
# "smalltalk_greetings_how_are_you"
# "smalltalk_greetings_nice_to_meet_you"
# "smalltalk_greetings_nice_to_see_you"
# "smalltalk_greetings_nice_to_talk_to_you"
# "smalltalk_greetings_whatsup"
# "smalltalk_user_angry"
# "smalltalk_user_back"
# "smalltalk_user_bored"
# "smalltalk_user_busy"
# "smalltalk_user_can_not_sleep"
# "smalltalk_user_does_not_want_to_talk"
# "smalltalk_user_excited"
# "smalltalk_user_going_to_bed"
# "smalltalk_user_good"
# "smalltalk_user_happy"
# "smalltalk_user_has_birthday"
# "smalltalk_user_here"
# "smalltalk_user_joking"
# "smalltalk_user_likes_agent"
# "smalltalk_user_lonely"
# "smalltalk_user_looks_like"
# "smalltalk_user_loves_agent"
# "smalltalk_user_misses_agent"
# "smalltalk_user_needs_advice"
# "smalltalk_user_sad"
# "smalltalk_user_sleepy"
# "smalltalk_user_testing_agent"
# "smalltalk_user_tired"
# "smalltalk_user_waits"
# "smalltalk_user_wants_to_see_agent_again"
# "smalltalk_user_wants_to_talk"
# "smalltalk_user_will_be_back"