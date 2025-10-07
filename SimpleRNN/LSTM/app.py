import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

model = load_model('next_word.h5')


with open('Tokenizer.pickle','rb') as file:
    token = pickle.load(file)

def predict_next_word(model,token,text,max_len):
    token_lists = token.texts_to_sequences([text])[0]
    if len(token_lists) >= max_len:
        token_lists = token_lists[-(max_len-1):]
    token_lists = pad_sequences([token_lists],maxlen=max_len-1,padding='pre')
    prediction = model.predict(token_lists, verbose=0)
    prediction_word_index = np.argmax(prediction,axis=1)
    for word,index in token.word_index.items():
        if index == prediction_word_index:
            return word
    return None


st.title("Next Word Prediction With Lstm and early Stopping")

input_text = st.text_input("Enter the firat Word")

if st.button("Predict Next Word"):
    max_seq = model.input_shape[1]+1
    next_word = predict_next_word(model,token,input_text,max_seq)
    st.write(f"The Next Word is {next_word}")

else:
    st.write("Enter the First word")