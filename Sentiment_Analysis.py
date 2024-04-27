import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_roc_curve
import pickle
from zipfile import ZipFile
import os
import matplotlib.pyplot as plt

# Download the Kaggle dataset
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kazanova/sentiment140

# Extract the dataset
with ZipFile('sentiment140.zip', 'r') as zip_ref:
    zip_ref.extractall()
    print('The dataset is extracted')

# Download NLTK stopwords
nltk.download('stopwords')

# Define column names for the dataset
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']

# Loading Twitter dataset
twitter_data = pd.read_csv('training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')

# Replacing target values to binary (0 for negative sentiment, 1 for positive sentiment)
twitter_data.replace({'target': {4: 1}}, inplace=True)

# Initializing Porter Stemmer for stemming
port_stem = PorterStemmer()

# Function for stemming the text data
def stemming(content):
    # Removing non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-z]', ' ', content)
    # Converting text to lowercase
    stemmed_content = stemmed_content.lower()
    # Tokenizing text
    stemmed_content = stemmed_content.split()
    # Stemming and removing stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # Joining tokens back into string
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content

# Applying stemming to text data
twitter_data['stemmed_content'] = twitter_data['text'].apply(stemming)

# Splitting the data into features (X) and target (Y)
X = twitter_data['stemmed_content'].values
Y = twitter_data['target'].values

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, stratify=Y, random_state=2)

# Initializing TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transforming text data into TF-IDF vectors
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Saving the TF-IDF vectorizer to disk
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)
    
# Initializing Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fitting the model on the training data
model.fit(X_train, Y_train)

# Predicting target labels for training data
X_train_prediction = model.predict(X_train)
# Calculating accuracy score for training data
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# Printing accuracy score for training data
print('Accuracy score of the training data:', training_data_accuracy)

# Predicting target labels for testing data
X_test_prediction = model.predict(X_test)
# Calculating accuracy score for testing data
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# Printing accuracy score for testing data
print('Accuracy score of the testing data:', testing_data_accuracy)

# Selecting a sample tweet for prediction
X_new = X_test[200]
# Corresponding actual sentiment label
print(Y_test[200])

# Predicting sentiment label for the sample tweet
prediction = model.predict(X_new)
# Printing predicted sentiment label
print(prediction)

# Printing the sentiment prediction
if prediction[0] == 0:
    print('Negative Tweet')
else:
    print('Positive Tweet')

# Plotting ROC Curve
plot_roc_curve(model, X_test, Y_test)
plt.title('ROC Curve')
plt.show()

# Saving the trained model to disk
filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))
