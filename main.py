# %%
from question_answer import answer_question
from small_talk import make_small_talk, replicate_answer
from match_intent import check_intent
from identity_management import set_username, is_name_change
from joblib import load

# %%
qa_data = load("qa_dataset.joblib")
sm_data = load("sm_dataset.joblib")
name_data = load("name_dataset.joblib")

# %%
name_data

#   (5 total features to pick from: 50% rule dictates that 3 features must be picked)
#
#   Core features:
#   Intent Matching: Use intent matching to distinguish whether something is a question or small talk
#   Question Answering: Check similarity of input to the dataset for q&a and then return the answer of that question.
#   Small talk: Check similarity of input to the dataset for small talk and then return the response for that input.
#
# %%
NAME = "NAME"
SMALLTALK = "SMALLTALK"
QUESTION = "QUESTION"

user_name = "User"

print("\nChatbot: Hi, I'm Steve. What is your name?\n")
user_input = input(f"{user_name}: ").lower()
user_name = set_username(user_input)
print(user_name)
print(
    f"Steve: Let me know at any time if you want to change your username {user_name} :)")
print("Steve: You can either chat with me or ask questions about the world.")

query = "TEMP STRING"
while (query):
    query = input(user_name + ": ").lower()
    intent_res = check_intent(name_data, sm_data, qa_data, query)

    intent = intent_res[0]
    similarity = intent_res[1]

    if intent == NAME:
        if (is_name_change(query)):
            print(f"Steve: {user_name}, please enter a new name")
            user_input = input(user_name + ": ").lower()
            user_name = set_username(user_input)
            print(f"Steve: Congratulations your name is now {user_name}")
        else:
            print(f"Steve: Your name is {user_name}")
    elif intent == SMALLTALK:
        response = make_small_talk(sm_data, similarity)
        if (response == "NOT FOUND"):
            new_response = replicate_answer(query)
            print("Steve: " + new_response)
        else:
            print("Steve: " + response)

    elif intent == QUESTION:
        response = answer_question(qa_data, similarity)

        if (response == "NOT FOUND"):
            print(
                "I'm sorry, I couldn't find what you are looking for. Can you try rephrasing the question?")
        else:
            print(f"Here is the answer to your question: {response}")

    else:
        print("I'm sorry, I didn't understand what you were saying.")


print("Steve: I enjoyed talking to you :)")

# %%
