import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import time

#Load data
house = pd.read_csv("ParisHousing.csv")

with st.spinner('Wait for it...'):
    time.sleep(5)

    st.title("Paris Housing Prices")

    #Show Data Frame
    option_sidebar = st.sidebar.selectbox("Select dataframe option", ('Hide', 'Show'))

    df = pd.DataFrame(house)
  
    # Displaying data frame with option on sidebar selection
    if option_sidebar == 'Show':
        st.header("Data Frame")
        st.write("This is the activity dataframe:")
        st.write(df.head(20))

    #Charts
    chart_data = pd.DataFrame(
        house,
        columns=['numPrevOwners', 'hasGuestRoom', 'cityPartRange']
    )

    third_chart_date = pd.DataFrame(
        house,
        columns=['made']
    )

    st.header("Chart Number One - Area Chart")
    st.area_chart(chart_data.head(20))

    st.header("Chart Number Two - Bar Chart")
    st.bar_chart(chart_data.head(20))

    st.header("Chart Number Three - Line Chart")
    st.line_chart(chart_data.head(20))

    #Start of ML Model
    #Sidebar
    st.sidebar.title("Select your parameters")
    st.sidebar.write("Navigate to change predictions")

    #Start of Prediction Parameters
    #Square Meters
    sm = st.sidebar.slider("Square Meters", 0, 100000, 50000)

    #Number of Rooms
    nr = st.sidebar.slider("How many rooms do you want?", 0, 150, 75)

    #Yard Selection
    yard = st.sidebar.radio("Do you want a yard?", ('Yes', 'No'))
    yard = 'Yes'
    if yard == 'No':
        hasYard = 0
    else:
        hasYard = 1

    #Pool Selection
    pool = st.sidebar.radio("Do you want a pool?", ('Yes', 'No'))
    pool = 'Yes'
    if pool == 'No':
        hasPool = 0
    else:
        hasPool = 1

    #Number of Floors
    fl = st.sidebar.slider("Number of floors", 0, 100, 50)

    #Year Made
    ym = st.sidebar.slider("Year Made", 1980, 2040, 2020)

    #Newly Built
    nb = st.sidebar.radio("Do you want a newly built house?", ('Yes', 'No'))
    nb = 'Yes'
    if nb == 'No':
        isNewBuilt = 0
    else:
        isNewBuilt = 1

    #Has Storm Protector
    sp = st.sidebar.radio("Do you want a storm protector?", ('Yes', 'No'))
    sp = 'Yes'
    if sp == 'No':
        hasStormProtector = 0
    else:
        hasStormProtector = 1

    #Has Storage Room
    sr = st.sidebar.radio("Do you want a storage room?", ('Yes', 'No'))
    sr = 'Yes'
    if sr == 'No':
        hasStorageRoom = 0
    else:
        hasStorageRoom = 1

    #Guest Room
    gr = st.sidebar.slider("How Many Guest Rooms", 0, 15, 7)

    #Main Body - Predictions
    #Main Body - Headings
    st.header("Machine Learning: Housing Price Predictions")
    st.write("A machine learning model for housing prices in Paris.")

    filename = 'finaloutput.sav'
    loaded_model = joblib.load(filename)
    prediction = round(loaded_model.predict([[sm, nr, hasYard, hasPool, fl, ym, isNewBuilt, hasStormProtector, hasStorageRoom, gr]])[0])
    st.write(f"Suggested Housing Price is: {prediction}")

    # Add a placeholder
    #latest_iteration = st.empty()
    #bar = st.progress(0)

    #for i in range(100):
        # Update the progress bar with each iterations
    #    latest_iteration.text(f"Iteration {i+1}")
    #    bar.progress(i+1)
    #    time.sleep(0.1)