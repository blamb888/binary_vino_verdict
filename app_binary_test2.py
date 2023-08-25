import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from st_files_connection import FilesConnection
import numpy as np
import os
import json

@st.cache_data
def load_model():
    # Create a FilesConnection object
    conn = st.experimental_connection('gcs', type=FilesConnection)

    # Assuming your model files are in a bucket named 'vino-verdict' in a folder called 'models'
    model_data = conn.read("vino-verdict/models/sentiment-bert-binary.bin")
    config_data = conn.read("vino-verdict/models/sentiment-bert-binary-config.json")

    # Create a local directory to store the model and config
    local_model_dir = './local_model'
    os.makedirs(local_model_dir, exist_ok=True)

    # Save the downloaded model and config to local directory
    with open(f"{local_model_dir}/pytorch_model.bin", 'wb') as f:
        f.write(model_data.read())

    with open(f"{local_model_dir}/config.json", 'w') as f:
        json.dump(json.loads(config_data.read().decode('utf-8')), f)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(local_model_dir)
    
    return model

# Define the conversion function
def convert_to_2_scale(arr):
    arr_2_scale = []
    for val in arr:
        if val in [0, 1]:
            arr_2_scale.append(0)  # bad
        else:
            arr_2_scale.append(1)  # average
    return np.array(arr_2_scale)

# Load the cached model
model = load_model()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Streamlit app
st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80", caption="Wine", use_column_width=True)
st.title('Binary Wine Sentiment Analysis')

user_input = st.text_area("Enter the review of the wine:")
if st.button('Predict'):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    predicted_classes = np.argmax(logits, axis=1)
    predicted_classes_2_scale = convert_to_2_scale(predicted_classes)  # convert to 2 scale
    
    verdict = 'good' if predicted_classes_2_scale[0] else 'bad'

    if verdict == 'good':
        st.markdown(f"<h1 style='text-align: center; color: green;'>This wine is: {verdict.upper()}</h1>", unsafe_allow_html=True)
        st.image("images/great_wine_wave.png", use_column_width=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: red;'>This wine is: {verdict.upper()}</h1>", unsafe_allow_html=True)
        st.image("images/bad_wine.png", use_column_width=True)
