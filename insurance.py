import streamlit as st
import numpy as np
import pandas as pd
import joblib
#import seaborn as sns

st.title("Insurance Pricing App")
st.write("From the insurance data, we build a machine learning model for pricing insurance claims.")

st.sidebar.title("Insurance Pricing App Parameters")
st.sidebar.write("Tweak to change predictions")

#Age
age = st.sidebar.slider("Age", 0, 100, 24)

#BMI
bmi = st.sidebar.slider("BMI", 15, 40, 29)

#Number of Children
num_children = st.sidebar.slider("Number of children", 0, 12, 1)

#Gender
gender = st.sidebar.radio("Gender", ('female', 'male'))
gender = 'female'
if gender == 'male':
    is_female = 0
else:
    is_female = 1

# Is Smoker
smoker = st.sidebar.radio("Smoker?" , ('yes', 'no'))
smoker = 'yes'
if smoker== 'yes':
    is_smoker = 1
else:
    is_smoker = 0
    
region = 'northwest'
if region == 'northeast':
    loc_list = [1, 0, 0, 0]
elif region == 'northwest':
    loc_list = [0, 1, 0, 0]
elif region == 'southeast':
    loc_list = [0, 0, 1, 0]
elif region == 'southwest':
    loc_list = [0, 0, 0, 1]

#Main Page
st.subheader("Predictions")

filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)
prediction = round(loaded_model.predict([[age, bmi, num_children, is_female, is_smoker] + loc_list])[0])
st.write(f"Suggested Insurance price is: {prediction}")