from small_talk import SmallTalk
from question_answer import QuestionAnswer
from identity_management import IdentityManagement
from joblib import load
from pprint import pprint
from process_text import preprocess_text

#   (5 total features to pick from: 50% rule dictates that 3 features must be picked)
#
#   Core features:
#   Intent Matching: Use intent matching to distinguish whether something is a question or small talk
#   Question Answering: Check similarity of input to the dataset for q&a and then return the answer of that question.
#   Small talk: Check similarity of input to the dataset for small talk and then return the response for that input.

NAME = "NAME"
SMALLTALK = "SMALL TALK"
QUESTION = "QUESTION"

small_talk = SmallTalk()
q_a = QuestionAnswer()
id_manage = IdentityManagement()

print("\nChatbot: Hi, I'm Steve. What is your name?\n")
user_input = input(f"User: ").lower()
user_name = id_manage.set_username(user_input)
print(
    f"Steve: Let me know at any time if you want to change your username {user_name} :)")
print("Steve: You can either chat with me or ask questions about:\n -- University, Youtube, Humanism, Geological History of Earth, Police, Infection, Hunting --\n")

query = "TEMP STRING"
incorrect_responses = 0
while (query):
    q_input = input(user_name + ": ")
    # If it is not question and answer. We give our chatbot a slightly more human response to give it more context.
    query = preprocess_text(text=q_input, type="lemmatisation")
    
    # Extracts only the relevant key words for the question and answer
    response = q_a.answer_question(query)
    if (response != "NOT FOUND"):
        print(f"Steve: Here is the answer to your question --- {response}")
        continue

    
    intent = id_manage.get_name_similarity(query)
    if intent == NAME:
        if (id_manage.is_name_change(query)):
            print(f"Steve: {user_name}, please enter a new name")
            user_input = input(user_name + ": ").lower()
            user_name = id_manage.set_username(user_input)
            print(f"Steve: Congratulations your name is now {user_name}")
            continue
        else:
            print(f"Steve: Your name is {user_name}")
            continue
    
    

    response = small_talk.find_response(query)
    if (response != "NOT FOUND"):
        print(f"Steve: {response}")
    else:
        if (incorrect_responses == 0):
            print("Steve: I'm sorry, I don't understand what you are saying. Could you try rephrasing?")
            incorrect_responses += 1
            continue
        elif (incorrect_responses >= 1):
            print("Steve: I'm really sorry. I don't have an answer. Can you find out and give me the answer so I can add it to my dataset.")
            incorrect_responses = 0
            continue    
    
    

print("Steve: I enjoyed talking to you :)")

# Issue I'm facing:
# I want the user to be able to add an answer if it isn't in the dataset for Q&A.
# Dilema    -- Chatbot works best when the Q&A is at the top.
#           -- Can't have an else condition for "NOT FOUND" because it means it will never check small talk dataset.
#           -- 
#           -- Can't move Q&A to the bottom.
#           -- It means for questions "What is youtube" they will be picked up as small talk.
#           -- Meaning short non-specific questions will register as small talk most of the time.
#           --
#           -- Can't add responses to small talk dataset because there is the UTTERANCE_COLUMN and INTENT_COLUM
#           -- I could fill the Utterance but I would have nothing for the intent_column.
#           -- Overly complicated, over engineering
# Simplest
# Answer    -- 1. Remove intents from Small Talk and just again have it as direct input leads to direct output.
#           -- ISSUE: I do not like this because it means that I do not have Intent Matching which is one of the features in the Coursework.
# 
#           -- 2. Maybe can somehow