import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import re
import pandas as pd
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
import time

# Set page configuration
st.set_page_config(
    page_title="Content Authenticity Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom font
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Header Styles */
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Result Styles */
    .result-real {
        font-size: 1.8rem;
        color: #10B981;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1FAE5;
        margin: 1rem 0;
    }
    
    .result-fake {
        font-size: 1.8rem;
        color: #EF4444;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FEE2E2;
        margin: 1rem 0;
    }
    
    .confidence {
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card Styles */
    .card {
        padding: 1.5rem;
        border-radius: 0.8rem;
        background-color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: rgba(255, 255, 255, 0.5);
        padding: 0.5rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 0.6rem;
        gap: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f1f5f9;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A !important;
        color: white !important;
        box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.3) !important;
    }
    
    /* Button Styles */
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(30, 58, 138, 0.2);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 6px 10px -1px rgba(30, 58, 138, 0.3);
        transform: translateY(-1px);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 4px -1px rgba(30, 58, 138, 0.2);
    }
    
    /* Progress Styles */
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background-color: #f1f5f9;
    }
    
    /* Info Box Styles */
    .info-box {
        background-color: #e0f2fe;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Statistics Card Styles */
    .stat-card {
        background-color: white;
        padding: 1.25rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.07), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
    }
    
    .stat-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #1E3A8A;
    }
    
    .stat-number {
        font-size: 2.25rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Dashboard Layout Styles */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Session State Initialization
# --------------------------------
if 'image_analyzed' not in st.session_state:
    st.session_state.image_analyzed = False
    
if 'text_analyzed' not in st.session_state:
    st.session_state.text_analyzed = False
    
if 'combined_analyzed' not in st.session_state:
    st.session_state.combined_analyzed = False
    
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
    
if 'text_result' not in st.session_state:
    st.session_state.text_result = None
    
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

if 'analysis_count' not in st.session_state:
    st.session_state.analysis_count = 0
    
if 'fake_detections' not in st.session_state:
    st.session_state.fake_detections = 0
    
if 'total_images' not in st.session_state:
    st.session_state.total_images = 0
    
if 'total_texts' not in st.session_state:
    st.session_state.total_texts = 0

# --------------------------------
# Helper Functions - Image Analysis
# --------------------------------
@st.cache_resource
def load_detection_model():
    """Load the image deepfake detection model."""
    try:
        return load_model('my_model.keras')
    except Exception as e:
        st.error(f"Error loading image model: {e}")
        return None

def predict_image(img_array, model):
    """Make predictions on the image using the model."""
    # Preprocess the image
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get class and confidence
    class_names = ['Fake', 'Real']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index]) * 100
    
    return predicted_class, confidence

# --------------------------------
# Helper Functions - Text Analysis
# --------------------------------
@st.cache_resource
def load_text_model(model_name):
    """Load a trained text analysis model from disk."""
    model_path = os.path.join('models', f"{model_name}_model.joblib")
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Model not found at {model_path}")
        return None

@st.cache_resource
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
    model = load_text_model(model_name)
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

# --------------------------------
# UI Components
# --------------------------------
def display_header():
    """Display the application header."""
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-header">Content Authenticity Analyzer</h1>
        <div style="display: flex; justify-content: center; gap: 1rem; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🔍</span>
                <span>Deepfake Detection</span>
            </div>
            <div style="color: #64748b;">|</div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">📝</span>
                <span>Text Analysis</span>
            </div>
            <div style="color: #64748b;">|</div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🛡️</span>
                <span>Content Protection</span>
            </div>
        </div>
        <p style="max-width: 700px; margin: 0 auto; color: #64748b; font-size: 1.1rem;">
            Detect deepfake images and fake social media content using advanced AI analysis.
            Stay protected in the era of synthetic media.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_dashboard():
    """Display the analysis dashboard with statistics."""
    st.markdown('<h2 class="sub-header">Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="margin-bottom: 1.5rem;">
        <h3 style="margin-top: 0;">Analysis Overview</h3>
        <p>Summary of all content analyses performed in this session.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">📊</div>
            <div class="stat-number">{}</div>
            <div class="stat-label">Total Analyses</div>
        </div>
        """.format(st.session_state.analysis_count), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">🖼️</div>
            <div class="stat-number">{}</div>
            <div class="stat-label">Images Analyzed</div>
        </div>
        """.format(st.session_state.total_images), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">📝</div>
            <div class="stat-number">{}</div>
            <div class="stat-label">Texts Analyzed</div>
        </div>
        """.format(st.session_state.total_texts), unsafe_allow_html=True)
    
    with col4:
        if st.session_state.analysis_count > 0:
            fake_percentage = (st.session_state.fake_detections / st.session_state.analysis_count) * 100
        else:
            fake_percentage = 0
            
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">⚠️</div>
            <div class="stat-number">{:.1f}%</div>
            <div class="stat-label">Fake Content Rate</div>
        </div>
        """.format(fake_percentage), unsafe_allow_html=True)
        
    # Display combined results if any analyses were done
    if st.session_state.image_analyzed or st.session_state.text_analyzed or st.session_state.combined_analyzed:
        st.markdown("""
        <div class="card" style="margin-top: 1.5rem;">
            <h3 style="margin-bottom: 1rem;">Latest Analysis Results</h3>
            <p>Summary of the most recent content analysis findings.</p>
        """, unsafe_allow_html=True)
        
        if st.session_state.combined_analyzed:
            # Display combined analysis results from simultaneous analysis
            img_result = st.session_state.image_result
            txt_result = st.session_state.text_result
            
            st.markdown("<h4>Combined Analysis</h4>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h5>Image Analysis</h5>", unsafe_allow_html=True)
                if img_result[0] == "Real":
                    st.markdown('<div style="font-size: 1.1rem; color: #10B981;">✅ Real Image</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size: 1.1rem; color: #EF4444;">⚠️ Deepfake</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.9rem;">Confidence: {img_result[1]:.1f}%</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown("<h5>Text Analysis</h5>", unsafe_allow_html=True)
                if txt_result[0] == "Real":
                    st.markdown('<div style="font-size: 1.1rem; color: #10B981;">✅ Authentic Content</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size: 1.1rem; color: #EF4444;">⚠️ Fake Content</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.9rem;">Confidence: {txt_result[1]:.1%}</div>', unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            # Determine the overall assessment
            if img_result[0] == "Fake" and txt_result[0] == "Fake":
                st.markdown("""
                <div class="result-fake" style="margin-top: 1rem;">⚠️ HIGH RISK: Both image and text appear to be fake</div>
                """, unsafe_allow_html=True)
            elif img_result[0] == "Fake" or txt_result[0] == "Fake":
                st.markdown("""
                <div class="result-fake" style="background-color: #FEF3C7; color: #D97706; margin-top: 1rem;">⚠️ MEDIUM RISK: Either image or text appears to be fake</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-real" style="margin-top: 1rem;">✅ LOW RISK: Both image and text appear to be authentic</div>
                """, unsafe_allow_html=True)
                
        # If not combined but both were analyzed separately
        elif st.session_state.image_analyzed and st.session_state.text_analyzed:
            # Get the results
            img_result = st.session_state.image_result
            txt_result = st.session_state.text_result
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h5>Image Analysis</h5>", unsafe_allow_html=True)
                if img_result[0] == "Real":
                    st.markdown('<div style="font-size: 1.1rem; color: #10B981;">✅ Real Image</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size: 1.1rem; color: #EF4444;">⚠️ Deepfake</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.9rem;">Confidence: {img_result[1]:.1f}%</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown("<h5>Text Analysis</h5>", unsafe_allow_html=True)
                if txt_result[0] == "Real":
                    st.markdown('<div style="font-size: 1.1rem; color: #10B981;">✅ Authentic Content</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="font-size: 1.1rem; color: #EF4444;">⚠️ Fake Content</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="font-size: 0.9rem;">Confidence: {txt_result[1]:.1%}</div>', unsafe_allow_html=True)
            
            st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
            
            # Determine the overall assessment
            if img_result[0] == "Fake" and txt_result[0] == "Fake":
                st.markdown("""
                <div class="result-fake" style="margin-top: 1rem;">⚠️ HIGH RISK: Both image and text appear to be fake</div>
                """, unsafe_allow_html=True)
            elif img_result[0] == "Fake" or txt_result[0] == "Fake":
                st.markdown("""
                <div class="result-fake" style="background-color: #FEF3C7; color: #D97706; margin-top: 1rem;">⚠️ MEDIUM RISK: Either image or text appears to be fake</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-real" style="margin-top: 1rem;">✅ LOW RISK: Both image and text appear to be authentic</div>
                """, unsafe_allow_html=True)
        
        # If only image was analyzed
        elif st.session_state.image_analyzed:
            img_result = st.session_state.image_result
            if img_result[0] == "Real":
                st.markdown("""
                <div class="result-real">✅ REAL IMAGE DETECTED</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-fake">⚠️ DEEPFAKE IMAGE DETECTED</div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"<div class='confidence'>Confidence: {img_result[1]:.2f}%</div>", unsafe_allow_html=True)
        
        # If only text was analyzed
        elif st.session_state.text_analyzed:
            txt_result = st.session_state.text_result
            prediction, confidence = txt_result[0], txt_result[1]
            
            if prediction == "Real":
                st.markdown("""
                <div class="result-real">✅ AUTHENTIC TEXT CONTENT</div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-fake">⚠️ FAKE TEXT CONTENT DETECTED</div>
                """, unsafe_allow_html=True)
                
            st.markdown(f"<div class='confidence'>Confidence: {confidence:.2%}</div>", unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

def display_image_analysis():
    """Display the image analysis tab content."""
    st.markdown('<h2 class="sub-header">Image Deepfake Detection</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        Upload an image to analyze it for potential manipulation or deepfake characteristics.
        Our AI model will assess the image and provide a determination with confidence score.
    </div>
    """, unsafe_allow_html=True)
    
    # Load the model
    model = load_detection_model()
    
    if model is None:
        st.warning("Please make sure the model file 'my_model.keras' is in the same directory as this script.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Process the uploaded image
    if uploaded_file is not None:
        try:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            
            # Display image with some styling
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(img, caption="Uploaded Image", use_column_width=True)
            
            # Add a button to run the detection
            if st.button("Analyze Image"):
                with st.spinner("Processing image..."):
                    # Add artificial delay for better UX
                    time.sleep(1)
                    
                    # Resize image to match model's expected input
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized)
                    
                    # If image has 4 channels (RGBA), convert to 3 channels (RGB)
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    
                    # Make prediction
                    predicted_class, confidence = predict_image(img_array, model)
                    
                    # Save results to session state
                    st.session_state.image_analyzed = True
                    st.session_state.image_result = (predicted_class, confidence)
                    st.session_state.analysis_count += 1
                    st.session_state.total_images += 1
                    
                    if predicted_class == "Fake":
                        st.session_state.fake_detections += 1
                    
                    # Display the result with appropriate styling
                    st.markdown("<h3 class='sub-header'>Detection Result:</h3>", unsafe_allow_html=True)
                    
                    if predicted_class == "Real":
                        st.markdown(f"<div class='result-real'>✅ REAL IMAGE</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-fake'>⚠️ DEEPFAKE DETECTED</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"<div class='confidence'>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
                    
                    # Display probability distribution
                    st.markdown("<h3 class='sub-header'>Probability Distribution:</h3>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(10, 3))
                    
                    # Create bar chart for probabilities
                    classes = ['Fake', 'Real']
                    probabilities = [
                        100 - confidence if predicted_class == "Real" else confidence,
                        confidence if predicted_class == "Real" else 100 - confidence
                    ]
                    
                    bars = ax.barh(
                        classes, 
                        probabilities, 
                        color=['#EF4444', '#10B981'],
                        height=0.5
                    )
                    
                    # Add percentage labels on bars
                    for bar, prob in zip(bars, probabilities):
                        ax.text(
                            min(prob + 3, 95), 
                            bar.get_y() + bar.get_height()/2, 
                            f'{prob:.1f}%', 
                            va='center', 
                            fontsize=12,
                            fontweight='bold',
                            color='black'
                        )
                    
                    ax.set_xlim(0, 100)
                    ax.set_xlabel('Probability (%)')
                    ax.set_title('Prediction Probabilities', fontsize=14, pad=10)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    
                    st.pyplot(fig)
                    
                    # Display explanation
                    with st.expander("What does this mean?"):
                        if predicted_class == "Real":
                            st.write("""
                            Our AI model has determined that this image is likely authentic and has not been manipulated 
                            using deepfake technology. The confidence score indicates how certain the model is about this assessment.
                            
                            However, no detection system is perfect. If you have reason to believe this image has been manipulated,
                            consider seeking a second opinion or examining metadata.
                            """)
                        else:
                            st.write("""
                            Our AI model has detected characteristics commonly associated with deepfake or manipulated images.
                            The confidence score indicates how certain the model is about this assessment.
                            
                            Deepfakes often have subtle inconsistencies in lighting, skin texture, facial features,
                            or background elements that our model can identify.
                            """)
                    
        except Exception as e:
            st.error(f"Error processing the image: {e}")

def display_text_analysis():
    """Display the text analysis tab content."""
    st.markdown('<h2 class="sub-header">Fake Content Text Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        Analyze text content from social media posts, comments, or messages to determine if it's likely from 
        a fake or authentic account. Our AI model evaluates language patterns and content markers.
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    model_options = {"Logistic Regression": "logistic", "Random Forest": "random_forest"}
    selected_model_name = st.selectbox("Select Analysis Model", list(model_options.keys()))
    model_type = model_options[selected_model_name]
    
    # Text input
    text_input = st.text_area(
        "Paste social media text here (post, comment, bio, etc.)",
        value=st.session_state.text_input,
        height=150
    )
    
    # Example button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Show Examples"):
            st.session_state['show_examples'] = not st.session_state.get('show_examples', False)
            
    # Analyze button
    with col2:
        analyze_clicked = st.button("Analyze Text", key="analyze_text_btn")
    
    # Show example texts if requested
    if st.session_state.get('show_examples', False):
        examples = get_example_texts()
        
        with st.expander("Example Texts (click to expand)", expanded=True):
            tab1, tab2 = st.tabs(["Real Account Examples", "Fake Account Examples"])
            
            with tab1:
                for i, example in enumerate(examples["real"]):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text_area(f"Example {i+1}", example, height=100, key=f"real_{i}")
                    with col2:
                        if st.button(f"Use", key=f"use_real_{i}"):
                            st.session_state.text_input = example
                            st.rerun()
            
            with tab2:
                for i, example in enumerate(examples["fake"]):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text_area(f"Example {i+1}", example, height=100, key=f"fake_{i}")
                    with col2:
                        if st.button(f"Use", key=f"use_fake_{i}"):
                            st.session_state.text_input = example
                            st.rerun()
    
    # Analyze text if requested
    if analyze_clicked and text_input:
        with st.spinner("Analyzing text..."):
            # Add artificial delay for better UX
            time.sleep(1)
            
            prediction, confidence, explanation = predict_account_type(text_input, model_type)
            
            # Save results to session state
            st.session_state.text_analyzed = True
            st.session_state.text_result = (prediction, confidence, explanation)
            st.session_state.analysis_count += 1
            st.session_state.total_texts += 1
            
            if prediction == "Fake":
                st.session_state.fake_detections += 1
        
        # Display result
        st.markdown("<h3 class='sub-header'>Analysis Result:</h3>", unsafe_allow_html=True)
        
        # Create columns for result display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display prediction with appropriate styling
            if prediction == "Real":
                st.markdown("<div class='result-real'>✅ AUTHENTIC CONTENT</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='result-fake'>⚠️ FAKE CONTENT DETECTED</div>", unsafe_allow_html=True)
            
            # Display confidence
            st.markdown(f"<div class='confidence'>Confidence: {confidence:.2%} ({explanation['confidence_level']})</div>", unsafe_allow_html=True)
            
            # Create a gauge chart for confidence
            fig, ax = plt.subplots(figsize=(8, 1))
            
            # Create gauge
            gauge_colors = ['#10B981', '#FBBF24', '#EF4444']
            bar_height = 0.4
            
            # Draw the gauge bar background
            ax.barh(0, 100, height=bar_height, color='#e5e7eb')
            
            # Draw the gauge value
            if prediction == "Real":
                value = (1 - confidence) * 100  # Invert for real predictions
            else:
                value = confidence * 100
                
            if value <= 33:
                color = gauge_colors[0]  # Green for low risk
            elif value <= 66:
                color = gauge_colors[1]  # Yellow for medium risk
            else:
                color = gauge_colors[2]  # Red for high risk
                
            ax.barh(0, value, height=bar_height, color=color)
            
            # Add marker for threshold
            ax.axvline(x=50, color='black', linestyle='--', alpha=0.7)
            
            # Add labels
            ax.text(0, -0.8, "Authentic", fontsize=10)
            ax.text(95, -0.8, "Fake", fontsize=10, ha='right')
            ax.text(50, -0.8, "Threshold", fontsize=10, ha='center')
            
            # Remove axes
            ax.set_ylim(-1, 1)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            st.pyplot(fig)
        
        with col2:
            # Display key indicators
            st.markdown("<h4>Key Indicators</h4>", unsafe_allow_html=True)
            if explanation['key_indicators']:
                for indicator in explanation['key_indicators']:
                    st.markdown(f"• {indicator}")
            else:
                st.markdown("No strong indicators detected.")
        
        # Display preprocessed text
        with st.expander("View Preprocessed Text"):
            st.code(explanation["preprocessed_text"])
        
        # Display information about fake accounts
        with st.expander("Learn More About Content Patterns"):
            if prediction == "Fake":
                st.markdown("""
                ### Common Traits of Fake Content:
                
                - **Excessive Capitalization**: USING ALL CAPS to grab attention
                - **Multiple Exclamation Marks**: Creating artificial urgency!!!
                - **Unrealistic Promises**: Offering improbable rewards or results
                - **Urgent Language**: Pressuring immediate action
                - **Request for Private Information**: Phishing for personal details
                - **Suspicious Links**: Directing to potentially harmful websites
                - **Poor Grammar/Spelling**: Lack of quality control or non-native writing
                - **Sensationalism**: Extreme claims designed to provoke emotional reactions
                """)
            else:
                st.markdown("""
                ### Common Traits of Authentic Content:
                
                - **Personal Pronouns**: Using "I," "me," "we," etc.
                - **Conversational Tone**: Natural, casual language
                - **Balanced Statements**: Avoiding extreme claims
                - **Questions and Engagement**: Seeking interaction with readers
                - **References to Daily Life**: Mentioning everyday activities
                - **Consistent Style**: Natural flow without erratic changes
                - **Appropriate Reactions**: Emotions that fit the context
                - **Specific Details**: Authentic personal experiences
                """)

def display_combined_analysis():
    """Display the combined analysis tab content."""
    st.markdown('<h2 class="sub-header">Combined Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0;">Comprehensive Content Analysis</h3>
        <p>Upload both an image and text for simultaneous analysis to get a complete assessment of content authenticity.</p>
        <p>This combined approach provides a more thorough evaluation by examining both visual and textual elements together.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout with two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">Upload Image</h3>', unsafe_allow_html=True)
        
        # Load the model
        model = load_detection_model()
        
        if model is None:
            st.warning("Please make sure the model file 'my_model.keras' is in the same directory as this script.")
            return
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="combined_image")
        
        # Display the uploaded image
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                img = Image.open(uploaded_file)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error displaying the image: {e}")
                uploaded_file = None
    
    with col2:
        st.markdown('<h3 class="sub-header">Enter Text</h3>', unsafe_allow_html=True)
        
        # Model selection
        model_options = {"Logistic Regression": "logistic", "Random Forest": "random_forest"}
        selected_model_name = st.selectbox("Select Text Analysis Model", list(model_options.keys()), key="combined_model")
        model_type = model_options[selected_model_name]
        
        # Text input
        text_input = st.text_area(
            "Paste social media text for analysis",
            value="",
            height=150,
            key="combined_text"
        )
    
    # Analyze button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("Analyze Both", key="analyze_combined_btn", use_container_width=True)
    
    # Process the analysis when button clicked
    if analyze_clicked:
        if uploaded_file is None and not text_input:
            st.warning("Please upload an image and enter text to analyze.")
        elif uploaded_file is None:
            st.warning("Please upload an image to analyze.")
        elif not text_input:
            st.warning("Please enter text to analyze.")
        else:
            with st.spinner("Analyzing content..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Process image
                img = Image.open(uploaded_file)
                progress_bar.progress(20)
                time.sleep(0.5)  # Small delay for better UX
                
                # Resize image to match model's expected input
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                
                # If image has 4 channels (RGBA), convert to 3 channels (RGB)
                if img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                
                # Make image prediction
                image_prediction, image_confidence = predict_image(img_array, model)
                progress_bar.progress(50)
                time.sleep(0.5)  # Small delay for better UX
                
                # Process text
                text_prediction, text_confidence, text_explanation = predict_account_type(text_input, model_type)
                progress_bar.progress(80)
                time.sleep(0.5)  # Small delay for better UX
                
                # Update session state
                st.session_state.combined_analyzed = True
                st.session_state.image_analyzed = True
                st.session_state.text_analyzed = True
                st.session_state.image_result = (image_prediction, image_confidence)
                st.session_state.text_result = (text_prediction, text_confidence, text_explanation)
                st.session_state.analysis_count += 1
                st.session_state.total_images += 1
                st.session_state.total_texts += 1
                
                if image_prediction == "Fake" or text_prediction == "Fake":
                    st.session_state.fake_detections += 1
                
                progress_bar.progress(100)
                
            # Display results
            st.markdown('<h3 class="sub-header">Analysis Results</h3>', unsafe_allow_html=True)
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h4>Image Analysis</h4>', unsafe_allow_html=True)
                
                if image_prediction == "Real":
                    st.markdown(f"<div class='result-real'>✅ REAL IMAGE</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-fake'>⚠️ DEEPFAKE DETECTED</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='confidence'>Confidence: {image_confidence:.2f}%</div>", unsafe_allow_html=True)
                
                # Display probability distribution
                fig, ax = plt.subplots(figsize=(8, 2))
                
                # Create bar chart for probabilities
                classes = ['Fake', 'Real']
                probabilities = [
                    100 - image_confidence if image_prediction == "Real" else image_confidence,
                    image_confidence if image_prediction == "Real" else 100 - image_confidence
                ]
                
                bars = ax.barh(
                    classes, 
                    probabilities, 
                    color=['#EF4444', '#10B981']
                )
                
                # Add percentage labels on bars
                for bar, prob in zip(bars, probabilities):
                    ax.text(
                        min(prob + 3, 95), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{prob:.1f}%', 
                        va='center', 
                        fontsize=12,
                        fontweight='bold'
                    )
                
                ax.set_xlim(0, 100)
                ax.set_xlabel('Probability (%)')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                st.pyplot(fig)
            
            with col2:
                st.markdown('<h4>Text Analysis</h4>', unsafe_allow_html=True)
                
                if text_prediction == "Real":
                    st.markdown("<div class='result-real'>✅ AUTHENTIC CONTENT</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-fake'>⚠️ FAKE CONTENT DETECTED</div>", unsafe_allow_html=True)
                
                st.markdown(f"<div class='confidence'>Confidence: {text_confidence:.2%} ({text_explanation['confidence_level']})</div>", unsafe_allow_html=True)
                
                # Display key indicators
                st.markdown("<h5>Key Indicators</h5>", unsafe_allow_html=True)
                if text_explanation['key_indicators']:
                    for indicator in text_explanation['key_indicators']:
                        st.markdown(f"• {indicator}")
                else:
                    st.markdown("No strong indicators detected.")
            
            # Overall assessment
            st.markdown('<h4 style="margin-top: 2rem;">Overall Assessment</h4>', unsafe_allow_html=True)
            
            if image_prediction == "Fake" and text_prediction == "Fake":
                st.markdown("""
                <div class="result-fake">⚠️ HIGH RISK: Both image and text appear to be fake</div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="margin-top: 1rem; padding: 1rem; background-color: #FEF2F2; border-radius: 0.5rem; border-left: 4px solid #EF4444;">
                    <h5 style="color: #B91C1C; margin-top: 0;">Warning</h5>
                    <p>This content shows strong indicators of being inauthentic in both visual and textual elements. 
                    Exercise extreme caution when encountering this type of content.</p>
                </div>
                """, unsafe_allow_html=True)
                
            elif image_prediction == "Fake" or text_prediction == "Fake":
                st.markdown("""
                <div class="result-fake" style="background-color: #FEF3C7; color: #D97706;">⚠️ MEDIUM RISK: Either image or text appears to be fake</div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="margin-top: 1rem; padding: 1rem; background-color: #FEF3C7; border-radius: 0.5rem; border-left: 4px solid #D97706;">
                    <h5 style="color: #B45309; margin-top: 0;">Caution</h5>
                    <p>This content contains elements that our analysis identifies as potentially manipulated or inauthentic.
                    Further verification is recommended before trusting or sharing.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown("""
                <div class="result-real">✅ LOW RISK: Both image and text appear to be authentic</div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="margin-top: 1rem; padding: 1rem; background-color: #D1FAE5; border-radius: 0.5rem; border-left: 4px solid #10B981;">
                    <h5 style="color: #047857; margin-top: 0;">Verified</h5>
                    <p>Based on our analysis, both the image and text content appear to be authentic.
                    As with all content, applying critical thinking is still recommended.</p>
                </div>
                """, unsafe_allow_html=True)




# --------------------------------
# Main Application
# --------------------------------
def main():
    """Main function to run the Streamlit application."""
    # Display header
    display_header()
    
    # Set up sidebar
    st.sidebar.markdown('<h2 class="sub-header">Tools & Settings</h2>', unsafe_allow_html=True)
    
    # Add tool information in sidebar
    st.sidebar.markdown("""
    <div class="info-box">
        <h4>🔍 Image Analysis</h4>
        <p>Upload photos to detect deepfakes using advanced AI.</p>
    </div>
    
    <div class="info-box">
        <h4>📝 Text Analysis</h4>
        <p>Analyze text content from social media for authenticity.</p>
    </div>
    
    <div class="info-box">
        <h4>🔄 Combined Analysis</h4>
        <p>Analyze both image and text simultaneously for a comprehensive assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button
    if st.sidebar.button("Reset All Analyses"):
        # Reset session state
        st.session_state.image_analyzed = False
        st.session_state.text_analyzed = False
        st.session_state.combined_analyzed = False
        st.session_state.image_result = None
        st.session_state.text_result = None
        st.session_state.text_input = ""
        st.rerun()
    
    # Display statistics in the sidebar
    st.sidebar.markdown('<h3 class="sub-header">Session Statistics</h3>', unsafe_allow_html=True)
    
    # Create metrics with icons
    col1, col2 = st.sidebar.columns(2)
    col1.metric("📊 Total Analyses", st.session_state.analysis_count)
    col2.metric("⚠️ Fake Detected", st.session_state.fake_detections)
    
    # Create a small chart for the sidebar
    if st.session_state.analysis_count > 0:
        # Create data for pie chart
        labels = ['Real', 'Fake']
        real_count = st.session_state.analysis_count - st.session_state.fake_detections
        sizes = [real_count, st.session_state.fake_detections]
        colors = ['#10B981', '#EF4444']
        
        # Create a pie chart
        fig, ax = plt.subplots(figsize=(4, 2.5))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=False, startangle=90,
                wedgeprops={'width': 0.5, 'edgecolor': 'w', 'linewidth': 2})
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.tight_layout()
        
        # Display the chart in the sidebar
        st.sidebar.pyplot(fig)
    
    # Information disclaimer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Disclaimer**: This application provides estimates based on AI models and should be used as a guide only. 
    No detection system is 100% accurate.
    """)
    
    # Create main tabs for different functionalities
    if st.session_state.image_analyzed or st.session_state.text_analyzed or st.session_state.combined_analyzed:
        # If analyses have been performed, show the dashboard first
        tabs = st.tabs(["Dashboard", "Image Analysis", "Text Analysis", "Combined Analysis"])
        
        with tabs[0]:
            display_dashboard()
            
        with tabs[1]:
            display_image_analysis()
            
        with tabs[2]:
            display_text_analysis()
            
        with tabs[3]:
            display_combined_analysis()
    else:
        # If no analyses yet, don't show dashboard tab
        tabs = st.tabs(["Image Analysis", "Text Analysis", "Combined Analysis"])
        
        with tabs[0]:
            display_image_analysis()
            
        with tabs[1]:
            display_text_analysis()
            
        with tabs[2]:
            display_combined_analysis()

# Run the application
if __name__ == "__main__":
    main()