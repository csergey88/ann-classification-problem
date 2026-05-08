import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model



# Load the trained model, scaler, and label encoder (one-hot encoder)
model = load_model('ann_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoded_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    one_hot_encoder_geo = pickle.load(f)    


## Streamlit app
st.title("Customer Churn Prediction")

# Input fields for user data
input_data = {}
input_data['CreditScore'] = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
input_data['Geography'] = st.selectbox("Geography", options=['France', 'Spain', 'Germany'])
input_data['Gender'] = st.selectbox("Gender", options=['Male', 'Female'])
input_data['Age'] = st.number_input("Age", min_value=18, max_value=100, value=30)
input_data['Tenure'] = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
input_data['Balance'] = st.number_input("Balance", min_value=0.0, value=10000.0)
input_data['NumOfProducts'] = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
input_data['HasCrCard'] = st.selectbox("Has Credit Card", options=[0, 1])
input_data['IsActiveMember'] = st.selectbox("Is Active Member", options=[0, 1])
input_data['EstimatedSalary'] = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# One-hot encode the 'Geography' column
geo_encoded = one_hot_encoder_geo.transform([[input_data['Geography']]]).toarray()  
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))


# Combine one-hot encoded features with the rest of the input data
input_df = pd.DataFrame([input_data])
input_df["Gender"] = label_encoder_gender.transform(input_df["Gender"])
input_df = pd.concat([input_df.drop(columns=['Geography']), geo_encoded_df], axis=1)

# Standardize the input data
input_df = scaler.transform(input_df)


# Predict the probability of churn
prediction = model.predict(input_df)
prediction_probability = prediction[0][0]

# Display the predicted probability of churn
st.subheader("Predicted Probability of Churn")
st.write(f"{prediction_probability:.4f}")

