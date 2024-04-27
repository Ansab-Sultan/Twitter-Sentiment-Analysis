# Twitter Sentiment Analysis

Twitter Sentiment Analysis is a Python project that leverages machine learning techniques to analyze sentiment in Twitter data. The project preprocesses text data, trains a logistic regression model, and evaluates its performance. Additionally, it includes a ROC curve plot to visualize the model's performance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Sample Prediction](#sample-prediction)
- [ROC Curve](#roc-curve)
- [Saving the Model](#saving-the-model)
- [Streamlit App](#streamlit-app)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Social media platforms like Twitter provide a wealth of data that can offer insights into public sentiment. Twitter Sentiment Analysis aims to extract meaningful sentiment from Twitter data by employing advanced machine learning algorithms.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/twitter-sentiment-analysis.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Navigate to the project directory.
2. Run the Python script:

    ```bash
    python twitter_sentiment_analysis.py
    ```

## Dataset

The Twitter dataset used in this project comprises tweets labeled with sentiment values (0 for negative sentiment, 1 for positive sentiment).

## Preprocessing

- Text data undergoes extensive preprocessing, including removal of non-alphabetic characters, conversion to lowercase, tokenization, and stemming.
- Stopwords are eliminated to focus on meaningful words.

## Model Training

- TF-IDF vectors are generated from preprocessed text data.
- A logistic regression model is trained using the TF-IDF vectors.

## Evaluation

- The model's accuracy is evaluated on both training and testing data.
- Performance metrics such as accuracy score are calculated.

## Sample Prediction

- A sample tweet is selected for prediction.
- The trained model predicts the sentiment label (positive/negative) for the sample tweet.

## ROC Curve

- A ROC curve plot is included to visualize the model's performance in terms of true positive rate versus false positive rate.

## Saving the Model

- The trained logistic regression model is serialized to disk using the pickle library for future use.

## Streamlit App

A Streamlit web application is included for real-time sentiment analysis. Users can input text and receive sentiment predictions instantly.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- streamlit

## Contributing

- [Muhammad Ansab Sultan](https://github.com/Ansab-Sultan) - Projects Developer

Contributions to this project are welcome! Feel free to contribute by opening issues or submitting pull requests.

## License

This project is licensed under the MIT License.

---

