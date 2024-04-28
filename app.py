import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
from pycaret.regression import setup, compare_models, pull, save_model, load_model

st.set_page_config(page_title="AutoVibhawML", page_icon="https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoVibhawML")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This application helps you build an ML model and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file)
        st.dataframe(df)
        df.to_csv('dataset.csv', index=False)

if choice == "Profiling":
    st.title("Auto EDA")
    profile = ProfileReport(df)
    st_profile_report(profile)

if choice == "Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
        
    if st.button('Start Training'):
        setup(df, target=chosen_target, silent=True)
        setup_df = pull().data.iloc[1:]
        st.dataframe(setup_df)
        save_model(compare_models(), 'best_model')
        st.dataframe(pull())


if choice == "Download the Model": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
