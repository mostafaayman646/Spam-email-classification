import streamlit as st
import pandas as pd
import pickle
from preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer

st.title("📩 Spam / Ham Classifier App")

try:
    model = pickle.load(open('Model/XGBoost_pipeline.pkl', 'rb'))
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning("⚠️ Model not available yet. Waiting for model.pkl file.")
    st.text(f"Error: {e}")

st.write("## 🔍 Try it yourself!")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message to classify.")
    elif not model_loaded:
        st.info("Model not loaded yet — please upload model.pkl first.")
    else:
        try:
            # Preprocess the input after checking it's not empty
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict([preprocessed_input])[0]

            if prediction == 1:
                st.error("🚫 SPAM Message")
            else:
                st.success("✅ HAM (Normal) Message")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")