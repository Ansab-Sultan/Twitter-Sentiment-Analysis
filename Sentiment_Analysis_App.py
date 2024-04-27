import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re

# Download NLTK stopwords
nltk.download('stopwords')

# Load the trained model
def load_model():
    with open('trained_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the TF-IDF vectorizer
def load_vectorizer():
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

# Function for preprocessing text
def preprocess_text(content):
    # Initializing Porter Stemmer for stemming
    port_stem = PorterStemmer()
    # Removing non-alphabetic characters
    preprocessed_content = re.sub('[^a-zA-z]', ' ', content)
    # Converting text to lowercase
    preprocessed_content = preprocessed_content.lower()
    # Tokenizing text
    preprocessed_content = preprocessed_content.split()
    # Stemming and removing stopwords
    preprocessed_content = [port_stem.stem(word) for word in preprocessed_content if not word in stopwords.words('english')]
    # Joining tokens back into string
    preprocessed_content = ' '.join(preprocessed_content)
    return preprocessed_content

# Function for predicting sentiment
def predict_sentiment(text, model, vectorizer):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    # Transform the preprocessed text into TF-IDF vector
    tfidf_vector = vectorizer.transform([preprocessed_text])
    # Predict the sentiment
    prediction = model.predict(tfidf_vector)
    return prediction[0]

# Title of the web app
st.title('Sentiment Analysis App')

# Load the trained model
model = load_model()

# Load the TF-IDF vectorizer
vectorizer = load_vectorizer()

# Input text box for user input
user_input = st.text_area("Enter your text here:")

if st.button('Analyze'):
    # Predict sentiment for the user input
    sentiment = predict_sentiment(user_input, model, vectorizer)
    if sentiment == 0:
        st.write('Negative Sentiment')
    else:
        st.write('Positive Sentiment')
