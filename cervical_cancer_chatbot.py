import streamlit as st
import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Sidebar contents.
import streamlit as st
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the BERT model and tokenizer
with st.sidebar:
    st.markdown('<h1 style="color:blue;">🛡️ Welcome to Shield Bot by Nayak! 🤖</h1>', unsafe_allow_html=True)
    st.header('About Nayak')

    st.markdown('''
    ## Empowering Reporting and Insights

    Nayak is a cutting-edge reporting and insights platform designed to empower individuals, particularly victims, to file reports openly or anonymously. The platform is committed to user convenience and offers morning and night modes.

    🚀 Once a report is submitted, Nayak provides authorized authorities with a comprehensive dashboard. Reports are intelligently categorized into open, in-progress, and closed statuses, ensuring efficient case management.

    ### AI-Powered Efficiency
    - Nayak seamlessly integrates AI to manage the influx of reports effectively.
    - The AI is adept at generating precise answers to specific questions posed by authorities, streamlining the information processing workflow.

    ### API Gateway for Data Insights
    - Nayak offers an API gateway, allowing authorities to access valuable data stored within the platform's databases.
    - Authorized external parties can retrieve pertinent information through the API, contributing to insights generation.
    

    💡 Note: The  Bot is an integral component of Nayak, augmenting the platform's capabilities and enhancing user experience.
    ''')
    st.markdown('<style>div.stNamedPlaceholder>div{margin-top:20px;}</style>', unsafe_allow_html=True)

# Download the punkt package
nltk.download('punkt', quiet=True)

# Load the dataset
import json

with open('dataset.json', 'r') as file:
    dataset = json.load(file)

# Function to return a random greeting respond to users' greeting
def greeting_response(text):
    text = text.lower()

    # Bot's greeting response
    bot_greetings = ['Hi there!', 'Hello', 'Heyy']
    # Users greetings
    user_greetings = ['hi', 'hey', 'hello', 'greetings']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)

def bot_response(user_input):
    # Preprocess user input
    user_input = user_input.lower()

    # Calculate similarity between user input and dataset questions
    similarity_scores = []
    for qa_pair in dataset:
        question = qa_pair['question'].lower()
        combined_text = [user_input, question]

        # Convert text to vectors
        vectorizer = CountVectorizer().fit_transform(combined_text)

        # Calculate cosine similarity
        similarity = cosine_similarity(vectorizer)

        # Append similarity score
        similarity_scores.append(similarity[0, 1])

    # Find the index of the most similar question
    max_index = np.argmax(similarity_scores)

    # Return the corresponding answer
    bot_response = dataset[max_index]['answer']

    return bot_response

import re
import random

st.title('Cervical Cancer Chatbot')

def handle_greeting(user_input):
    greetings = ['hello', 'hi', 'hey', 'howdy', 'hola', 'hii']
    for word in user_input.split():
        if word.lower() in greetings:
            return random.choice(greetings)

exit_list = ['exit', 'see you later', 'bye', 'quit', 'break', 'done']

# Initialize chat history
chat_history = []

user_input = st.text_input("You: ")
if user_input.lower() in exit_list:
    st.write('Chat with you later!')
else:
    greeting = handle_greeting(user_input)
    if greeting is not None:
        st.write( greeting + '. How can I assist you today?')
        chat_history.append(( user_input))
        chat_history.append(( greeting + '. How can I assist you today?'))
    else:
        bot_resp = bot_response(user_input)
        st.write(bot_resp)
        chat_history.append(( user_input))
        chat_history.append(( bot_resp))

# Display chat history
st.write('')
