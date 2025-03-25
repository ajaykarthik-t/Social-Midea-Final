#!/usr/bin/env python3
"""
app.py - Simple Streamlit app for fake social media account detection.
"""

import os
import json
import re
import numpy as np
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer


def load_model(model_name):
    """Load a trained model from disk."""
    model_path = os.path.join('models', f"{model_name}_model.joblib")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model not found at {model_path}")
        return None


def load_vectorizer():
    """Load the vectorizer from disk."""
    vectorizer_path = os.path.join('data', "vectorizer.json")
    
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'r') as f:
            vectorizer_data = json.load(f)
        
        vectorizer = CountVectorizer()
        vectorizer.vocabulary_ = vectorizer_data['vocabulary']
        
        return vectorizer
    else:
        st.error(f"Vectorizer not found at {vectorizer_path}")
        return None


def preprocess_text(text):
    """Basic text preprocessing."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\S+', '', text)
    
    # Remove hashtags as symbols but keep the text
    text = re.sub(r'#(\S+)', r'\1', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_features(text, vectorizer):
    """Extract features from text using the vectorizer."""
    # Preprocess text
    preprocessed_text = preprocess_text(text)
    
    # Transform text to features
    features = vectorizer.transform([preprocessed_text]).toarray()
    
    return features, preprocessed_text


def predict_account_type(text, model_name='logistic'):
    """
    Predict if a social media account is fake or real based on text.
    
    Returns:
        Tuple of (prediction, confidence, explanation)
    """
    # Load model and vectorizer
    model = load_model(model_name)
    vectorizer = load_vectorizer()
    
    if model is None or vectorizer is None:
        return "Error", 0.0, {"error": "Failed to load model or vectorizer"}
    
    # Extract features
    features, preprocessed_text = extract_features(text, vectorizer)
    
    # Make prediction
    prediction_proba = model.predict_proba(features)[0]
    fake_confidence = prediction_proba[1]
    
    # Determine prediction
    if fake_confidence >= 0.5:
        prediction = "Fake"
    else:
        prediction = "Real"
    
    # Generate explanation
    explanation = {
        "original_text": text,
        "preprocessed_text": preprocessed_text,
        "confidence": fake_confidence,
        "confidence_level": get_confidence_level(fake_confidence),
        "key_indicators": get_key_indicators(text, prediction)
    }
    
    return prediction, fake_confidence, explanation


def get_confidence_level(confidence):
    """Convert numerical confidence to text description."""
    # Adjust confidence to be relative to the decision boundary (0.5)
    adjusted_conf = abs(confidence - 0.5) * 2  # Scale to 0-1
    
    if adjusted_conf < 0.2:
        return "Very low"
    elif adjusted_conf < 0.4:
        return "Low"
    elif adjusted_conf < 0.6:
        return "Moderate"
    elif adjusted_conf < 0.8:
        return "High"
    else:
        return "Very high"


def get_key_indicators(text, prediction):
    """Identify key indicators in the text based on the prediction."""
    text_lower = text.lower()
    
    fake_indicators = [
        ("CAPITAL LETTERS", "ALL CAPS" in text or text.isupper()),
        ("Multiple exclamation marks", "!!" in text),
        ("Money symbols", "$" in text or "€" in text),
        ("Urgency words", any(word in text_lower for word in ["urgent", "hurry", "limited time", "act now"])),
        ("Free offers", "free" in text_lower and any(word in text_lower for word in ["win", "gift", "offer", "giveaway"]))
    ]
    
    real_indicators = [
        ("Personal pronouns", any(word in text_lower for word in [" i ", " me ", " my ", " mine ", " we ", " our "])),
        ("Conversational tone", any(word in text_lower for word in ["thanks", "thank you", "appreciate", "hope", "think"])),
        ("Questions", "?" in text),
        ("Everyday activities", any(word in text_lower for word in ["today", "yesterday", "tomorrow", "weekend", "week"]))
    ]
    
    if prediction == "Fake":
        return [ind for ind, present in fake_indicators if present]
    else:
        return [ind for ind, present in real_indicators if present]


def get_example_texts():
    """Return example texts for demonstration."""
    real_examples = [
        "Just finished hiking with my family. The views were breathtaking! Can't wait to share the photos.",
        "Anyone have recommendations for a good book to read this weekend? I just finished my last one.",
        "Spent the afternoon baking cookies with my daughter. They didn't turn out perfect, but we had fun!",
        "Feeling a bit under the weather today. Going to make some tea and take it easy.",
        "So excited to announce that I've been accepted to graduate school! Starting this fall."
    ]
    
    fake_examples = [
        "MAKE $5000 FROM HOME IN JUST 2 HOURS!!! Click this link to learn my secret method: bit.ly/not-a-scam",
        "BREAKING NEWS: Famous celebrity reveals shocking government secrets! This is being censored everywhere!",
        "I lost 50 pounds in just one week using this miracle pill! Doctors HATE this! Click here: tinyurl.com/fake-diet",
        "FREE iPhone giveaway!! Apple is giving away 100 phones today only! Just retweet and click to register!",
        "URGENT: Your account will be suspended in 24 hours! Verify your information now by sending your login details!"
    ]
    
    return {"real": real_examples, "fake": fake_examples}


def main():
    """Main function to run the Streamlit app."""
    # Set page title
    st.set_page_config(
        page_title="Fake Account Detector",
        page_icon="🕵️"
    )
    
    # Display header
    st.title("Fake Social Media Account Detector")
    st.write("This application uses machine learning to analyze text and determine if it's likely from a real or fake social media account.")
    
    # Sidebar options
    st.sidebar.header("Options")
    
    # Model selection
    model_options = {"Logistic Regression": "logistic", "Random Forest": "random_forest"}
    selected_model_name = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_type = model_options[selected_model_name]
    
    # About section
    st.sidebar.header("About")
    st.sidebar.info("""
    This application detects fake social media accounts based on text analysis.
    
    It uses machine learning models trained on patterns found in real vs. fake accounts.
    """)
    
    # Example texts
    if st.sidebar.button("Show Example Texts"):
        st.session_state['show_examples'] = not st.session_state.get('show_examples', False)
    
    # Show example texts if requested
    if st.session_state.get('show_examples', False):
        st.header("Example Texts")
        
        examples = get_example_texts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Real Account Examples")
            for i, example in enumerate(examples["real"]):
                st.text_area(f"Real Example {i+1}", example, height=100, key=f"real_{i}")
                if st.button(f"Use this example", key=f"use_real_{i}"):
                    st.session_state['text_input'] = example
                    st.rerun()
        
        with col2:
            st.subheader("Fake Account Examples")
            for i, example in enumerate(examples["fake"]):
                st.text_area(f"Fake Example {i+1}", example, height=100, key=f"fake_{i}")
                if st.button(f"Use this example", key=f"use_fake_{i}"):
                    st.session_state['text_input'] = example
                    st.rerun()
    
    # Main content
    st.header("Enter Social Media Text")
    
    # Initialize session state for text input
    if 'text_input' not in st.session_state:
        st.session_state['text_input'] = ""
    
    # Text input
    text_input = st.text_area(
        "Paste social media text here (post, comment, bio, etc.)",
        value=st.session_state['text_input'],
        height=150
    )
    
    # Clear session state after use
    st.session_state['text_input'] = ""
    
    # Analyze button
    if st.button("Analyze Text") or text_input:
        if not text_input:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                prediction, confidence, explanation = predict_account_type(text_input, model_type)
            
            # Display result
            st.header("Analysis Result")
            
            # Create columns for result display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display prediction with appropriate styling
                if prediction == "Real":
                    st.success(f"**Prediction:** {prediction} Account")
                elif prediction == "Fake":
                    st.error(f"**Prediction:** {prediction} Account")
                else:
                    st.warning(f"**Prediction:** {prediction}")
                
                # Display confidence
                st.info(f"**Confidence:** {confidence:.2%}")
                st.write(f"**Confidence Level:** {explanation['confidence_level']}")
            
            with col2:
                # Display key indicators
                st.subheader("Key Indicators")
                if explanation['key_indicators']:
                    for indicator in explanation['key_indicators']:
                        st.write(f"- {indicator}")
                else:
                    st.write("No strong indicators detected.")
            
            # Display preprocessed text
            with st.expander("View Preprocessed Text"):
                st.text(explanation["preprocessed_text"])
            
            # Display information about fake accounts
            if prediction == "Fake":
                with st.expander("Common Fake Account Characteristics"):
                    st.markdown("""
                    ### Common Traits of Fake Accounts:
                    - Excessive use of capital letters
                    - Multiple exclamation marks
                    - Offering unrealistic rewards or deals
                    - Requests for personal information
                    - Suspicious links
                    - Urgent calls to action
                    - Poor grammar and spelling
                    """)


if __name__ == "__main__":
    main()