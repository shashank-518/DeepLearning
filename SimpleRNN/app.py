import pandas as pd
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential,load_model
from keras.layers import Dense,Embedding,SimpleRNN
from keras.preprocessing import sequence
import streamlit as st

word_index = imdb.get_word_index()
reverse_words = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def movieReview(rev):
    processed_input = preprocess_text(rev)

    prediction = model.predict(processed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment,prediction[0][0]

st.title("Sentiment of Movie")
st.write('Enter a movie review to classify it as positive or negative')

input_text = st.text_area('Movie Review')

if st.button('Classify'):
    

    setiment,pred = movieReview(input_text)

    st.write(f"The Movie predition is:")
    st.write(f"The Setiment of the movie is: {setiment}")
    st.write(f"The prediction of positive/negative is: {pred}")


else:
    st.write('Please write Movie Reviews')