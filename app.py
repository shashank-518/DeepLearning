import streamlit as st
import pandas as pd
import numpy as np 
import pickle
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import tensorflow as tf
import keras as k

model = k.models.load_model('model.h5')


with open('label_encoder_gender.pkl', mode='rb') as file:
    label_encoder_gender = pickle.load(file)

with open('oneHot_geo.pkl',mode='rb') as file:
    oneHot_geo = pickle.load(file)

with open('standard_scaler.pkl', mode='rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', oneHot_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=900, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}


geo_encoded = oneHot_geo.transform([[geography]]).toarray()
geo_encoded = pd.DataFrame(geo_encoded,columns=oneHot_geo.get_feature_names_out(['Geography']))



input_df = pd.DataFrame([input_data])
final_input = pd.concat([input_df.reset_index(drop=True), geo_encoded.reset_index(drop=True)], axis=1)

input_scaler = scaler.transform(final_input)

prediction_proba = model.predict(input_scaler)[0][0]
if prediction_proba > 0.5:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
st.write(f'Churn probability: {prediction_proba:.2f}')