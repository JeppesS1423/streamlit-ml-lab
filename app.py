import streamlit as st
import pandas as pd
import json
import redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host="redis", decode_responses=True)

st.title("Machine Learning Lab")
st.write("Try out different machine learning models on datasets.\n\nBegin by uploading your dataset, or try one of the sample ones!\n\n")

if "data_choice" not in st.session_state:
    st.session_state.data_choice = None

# Dataset choice buttons
data_col1, data_col2, data_col3, data_col4, data_col5 = st.columns(5)

if data_col1.button("Iris", use_container_width=True):
    st.session_state.data_choice = "iris"
    data = r.get("dataset:iris")
    if data is None:
        data = pd.read_csv("data/sample_datasets/Iris.csv")
        r.set("dataset:iris", data.to_json())

if data_col2.button("Boston", use_container_width=True):
    st.session_state.data_choice = "boston"
    data = r.get("dataset:boston")
    if data is None:
        data = pd.read_csv("data/sample_datasets/boston.csv")
        r.set("dataset:boston", data.to_json())

if data_col3.button("Sonar", use_container_width=True):
    st.session_state.data_choice = "sonar"
    data = r.get("dataset:sonar")
    if data is None:
        data = pd.read_csv("data/sample_datasets/sonar_data.csv")
        r.set("dataset:sonar", data.to_json())

if data_col4.button("Insurance", use_container_width=True):
    st.session_state.data_choice = "insurance"
    data = r.get("dataset:insurance")
    if data is None:
        data = pd.read_csv("data/sample_datasets/swedish_auto_insurance.csv")
        r.set("dataset:insurance", data.to_json())

if data_col5.button("Upload", use_container_width=True):
    st.session_state.data_choice = "upload"

# Dataset info container
with st.container(border=True):
    if st.session_state.data_choice == "iris":
        st.header("Iris")
        data = r.get("dataset:iris")
        data = pd.read_json(data)
        st.write(data.head(5))
        st.write(data.describe())

    elif st.session_state.data_choice == "boston":
        st.header("Boston House Prices")
        data = r.get("dataset:boston")
        data = pd.read_json(data)
        st.write(data.head(5))
        st.write(data.describe())

    elif st.session_state.data_choice == "sonar":
        st.header("Sonar Data")
        data = r.get("dataset:sonar")
        data = pd.read_json(data)
        st.write(data.head(5))
        st.write(data.describe())

    elif st.session_state.data_choice == "insurance":
        st.header("Swedish Auto Insurance")
        data = r.get("dataset:insurance")
        data = pd.read_json(data)
        st.write(data.head(5))
        st.write(data.describe())

    elif st.session_state.data_choice == "upload":
        st.header("Upload Your Data")