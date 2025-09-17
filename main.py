
##Step1 : load all the libraries
import numpy as np 
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model


##Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key, value in word_index.items()}

## Load the pretrained model with Relu index
model = load_model('simple_rnn_imdb.h5')


#Step 2 : Helper function
##Function to decode the reviews
def decode_review(encoded_review):
    return  ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

##Function to preprocess the user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Step 3 : create prediction function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment,prediction[0][0]


##Streamlit app
import streamlit as st

st.title('IMDB review Sentiment analysis')
st.write('Enter a moveie name to classify it as positive or negative')

##User input
user_input = st.text_area('Move review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    ##Make prediction
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    ##Display the result
    st.write(f'Sentiment : {sentiment}')
    st.write(f'Prediction score : {prediction[0][0]}')
else:
    st.write('Please enter the movie review.')