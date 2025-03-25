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
    
if 'account_analyzed' not in st.session_state:
    st.session_state.account_analyzed = False
    
if 'image_result' not in st.session_state:
    st.session_state.image_result = None
    
if 'text_result' not in st.session_state:
    st.session_state.text_result = None
    
if 'account_result' not in st.session_state:
    st.session_state.account_result = None
    
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

if 'total_accounts' not in st.session_state:
    st.session_state.total_accounts = 0

if 'total_risk_score' not in st.session_state:
    st.session_state.total_risk_score = 0

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
# Helper Functions - Account Analysis
# --------------------------------
def analyze_account_metrics(
    account_age, 
    follower_count, 
    following_count, 
    post_count, 
    engagement_rate, 
    profile_completeness,
    posting_frequency,
    is_verified,
    content_consistency,
    has_external_links,
    geo_consistency
):
    """
    Analyze social media account metrics to determine authenticity.
    
    Returns:
        tuple: (prediction, metrics_dict)
    """
    # Initialize metrics dictionary
    metrics = {
        'confidence': 0.0,
        'risk_score': 0.0,
        'positive_indicators': {},
        'suspicious_indicators': {},
        'recommendations': []
    }
    
    # Calculate follower-to-following ratio (if applicable)
    follower_ratio = follower_count / max(following_count, 1)
    
    # Calculate post frequency (posts per day)
    post_frequency = post_count / max(account_age, 1)
    
    # Initialize risk score (0-100, higher means more suspicious)
    risk_score = 50  # Start at neutral
    
    # --- Analyze account age ---
    if account_age < 30:
        risk_score += 15
        metrics['suspicious_indicators']['New Account'] = f"{account_age} days old"
        metrics['recommendations'].append("New accounts with high following/follower counts are often suspicious")
    else:
        risk_score -= 10
        metrics['positive_indicators']['Established Account'] = f"{account_age} days old"
    
    # --- Analyze follower-to-following ratio ---
    if follower_count > 1000 and follower_ratio < 0.1:
        risk_score += 15
        metrics['suspicious_indicators']['Imbalanced Following Ratio'] = f"{follower_ratio:.2f}"
        metrics['recommendations'].append("Accounts following many users with few followers can indicate automation")
    elif follower_ratio > 0.5:
        risk_score -= 10
        metrics['positive_indicators']['Healthy Following Ratio'] = f"{follower_ratio:.2f}"
    
    # --- Analyze engagement rate ---
    if follower_count > 1000 and engagement_rate < 1.0:
        risk_score += 20
        metrics['suspicious_indicators']['Low Engagement'] = f"{engagement_rate:.1f}%"
        metrics['recommendations'].append("Low engagement despite high follower count suggests purchased followers")
    elif engagement_rate > 3.0:
        risk_score -= 15
        metrics['positive_indicators']['Strong Engagement'] = f"{engagement_rate:.1f}%"
    
    # --- Analyze post consistency ---
    if post_count < 5 and follower_count > 1000:
        risk_score += 20
        metrics['suspicious_indicators']['Few Posts, Many Followers'] = f"{post_count} posts"
        metrics['recommendations'].append("High follower count with minimal content is a red flag")
    
    if posting_frequency > 0 and posting_frequency < 20:
        risk_score -= 5
        metrics['positive_indicators']['Regular Posting Schedule'] = f"{posting_frequency} posts/week"
    elif posting_frequency > 50:
        risk_score += 10
        metrics['suspicious_indicators']['Excessive Posting'] = f"{posting_frequency} posts/week"
        metrics['recommendations'].append("Extremely high posting frequency can indicate automation")
    
    # --- Analyze profile completeness ---
    if profile_completeness < 30:
        risk_score += 10
        metrics['suspicious_indicators']['Incomplete Profile'] = f"{profile_completeness}% complete"
        metrics['recommendations'].append("Complete your profile to improve credibility")
    elif profile_completeness > 80:
        risk_score -= 10
        metrics['positive_indicators']['Complete Profile'] = f"{profile_completeness}% complete"
    
    # --- Analyze verification status ---
    if is_verified:
        risk_score -= 25
        metrics['positive_indicators']['Verified Account'] = "Yes"
    
    # --- Analyze content consistency ---
    if content_consistency < 3:
        risk_score += 10
        metrics['suspicious_indicators']['Inconsistent Content'] = f"{content_consistency}/10"
    elif content_consistency > 7:
        risk_score -= 10
        metrics['positive_indicators']['Consistent Content'] = f"{content_consistency}/10"
    
    # --- Analyze external links ---
    if has_external_links and account_age < 30 and follower_count > 5000:
        risk_score += 10
        metrics['suspicious_indicators']['New Account with Links'] = "Suspicious pattern"
        metrics['recommendations'].append("New accounts with links and many followers warrant caution")
    elif has_external_links and account_age > 180:
        risk_score -= 5
        metrics['positive_indicators']['Established with Web Presence'] = "Yes"
    
    # --- Analyze geographic consistency ---
    if geo_consistency < 3:
        risk_score += 5
        metrics['suspicious_indicators']['Geographic Inconsistency'] = f"{geo_consistency}/10"
    elif geo_consistency > 7:
        risk_score -= 5
        metrics['positive_indicators']['Geographic Consistency'] = f"{geo_consistency}/10"
    
    # Ensure risk score stays within 0-100 range
    risk_score = max(0, min(100, risk_score))
    metrics['risk_score'] = risk_score
    
    # Determine prediction based on risk score
    if risk_score < 40:
        prediction = "Authentic"
        confidence = 100 - risk_score
    elif risk_score < 70:
        prediction = "Attention"
        confidence = 100 - abs(risk_score - 50)
    else:
        prediction = "Suspicious"
        confidence = risk_score
    
    metrics['confidence'] = confidence
    
    # Add general recommendations
    if len(metrics['recommendations']) == 0:
        metrics['recommendations'].append("Continue monitoring account behavior for changes")
        metrics['recommendations'].append("Verify account through other communication channels")
    
    return prediction, metrics

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
                <span style="font-size: 1.5rem;">👤</span>
                <span>Account Analysis</span>
            </div>
            <div style="color: #64748b;">|</div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">🛡️</span>
                <span>Content Protection</span>
            </div>
        </div>
        <p style="max-width: 700px; margin: 0 auto; color: #64748b; font-size: 1.1rem;">
            Detect deepfake images, fake social media content, and suspicious account patterns using advanced AI analysis.
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
    
    # Add account analyses stat
    with col4:
        account_analyses = st.session_state.get('total_accounts', 0)
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">👤</div>
            <div class="stat-number">{}</div>
            <div class="stat-label">Accounts Analyzed</div>
        </div>
        """.format(account_analyses), unsafe_allow_html=True)
    
    # Second row of stats
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
        # Calculate average risk score if we have account analyses
        if st.session_state.get('total_accounts', 0) > 0 and hasattr(st.session_state, 'total_risk_score'):
            avg_risk = st.session_state.total_risk_score / st.session_state.total_accounts
        else:
            avg_risk = 0
            
        st.markdown("""
        <div class="stat-card">
            <div class="stat-icon">🛡️</div>
            <div class="stat-number">{:.1f}</div>
            <div class="stat-label">Avg. Account Risk Score</div>
        </div>
        """.format(avg_risk), unsafe_allow_html=True)
    
    # Display combined results if any analyses were done
    if st.session_state.image_analyzed or st.session_state.text_analyzed or st.session_state.get('account_analyzed', False) or st.session_state.combined_analyzed:
        st.markdown("""
        <div class="card" style="margin-top: 1.5rem;">
            <h3 style="margin-bottom: 1rem;">Latest Analysis Results</h3>
            <p>Summary of the most recent content analysis findings.</p>
        """, unsafe_allow_html=True)
        
        # Create tabs for different analysis types
        result_tabs = st.tabs(["Image", "Text", "Account", "Combined"])
        
        with result_tabs[0]:
            if st.session_state.image_analyzed:
                img_result = st.session_state.image_result
                if img_result[0] == "Real":
                    st.markdown('<div class="result-real">✅ REAL IMAGE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">⚠️ DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">Confidence: {img_result[1]:.2f}%</div>', unsafe_allow_html=True)
            else:
                st.markdown("No image analysis performed yet.")
        
        with result_tabs[1]:
            if st.session_state.text_analyzed:
                txt_result = st.session_state.text_result
                if txt_result[0] == "Real":
                    st.markdown('<div class="result-real">✅ AUTHENTIC CONTENT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">⚠️ FAKE CONTENT DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">Confidence: {txt_result[1]:.2%}</div>', unsafe_allow_html=True)
            else:
                st.markdown("No text analysis performed yet.")
        
        with result_tabs[2]:
            if st.session_state.get('account_analyzed', False):
                acc_result = st.session_state.account_result
                if acc_result[0] == "Authentic":
                    st.markdown('<div class="result-real">✅ LIKELY AUTHENTIC ACCOUNT</div>', unsafe_allow_html=True)
                elif acc_result[0] == "Attention":
                    st.markdown('<div class="result-fake" style="background-color: #FEF3C7; color: #D97706;">⚠️ REQUIRES ATTENTION</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">⚠️ SUSPICIOUS ACCOUNT DETECTED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">Risk Score: {acc_result[1]["risk_score"]:.1f}/100</div>', unsafe_allow_html=True)
                
                # Show key indicators
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Positive Indicators:**")
                    for i, (key, value) in enumerate(acc_result[1]["positive_indicators"].items()):
                        if i < 3:  # Show top 3
                            st.markdown(f"• {key}: {value}")
                
                with col2:
                    st.markdown("**Top Suspicious Indicators:**")
                    for i, (key, value) in enumerate(acc_result[1]["suspicious_indicators"].items()):
                        if i < 3:  # Show top 3
                            st.markdown(f"• {key}: {value}")
            else:
                st.markdown("No account analysis performed yet.")
                
        with result_tabs[3]:
            if st.session_state.combined_analyzed:
                # Display combined analysis results from simultaneous analysis
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
                
                # Show account analysis if available
                if st.session_state.get('account_analyzed', False):
                    st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                    st.markdown("<h5>Account Analysis</h5>", unsafe_allow_html=True)
                    
                    acc_result = st.session_state.account_result
                    if acc_result[0] == "Authentic":
                        st.markdown('<div style="font-size: 1.1rem; color: #10B981;">✅ Authentic Account</div>', unsafe_allow_html=True)
                    elif acc_result[0] == "Attention":
                        st.markdown('<div style="font-size: 1.1rem; color: #D97706;">⚠️ Requires Attention</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size: 1.1rem; color: #EF4444;">⚠️ Suspicious Account</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 0.9rem;">Risk Score: {acc_result[1]["risk_score"]:.1f}/100</div>', unsafe_allow_html=True)
                
                st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
                
                # Determine the overall assessment incorporating all available analyses
                overall_risk = "LOW"
                risk_color = "#10B981"  # Green
                risk_message = "Content appears authentic across all analysis types"
                
                # Check image and text
                if img_result[0] == "Fake" and txt_result[0] == "Fake":
                    overall_risk = "HIGH"
                    risk_color = "#EF4444"  # Red
                    risk_message = "Both image and text appear to be fake"
                elif img_result[0] == "Fake" or txt_result[0] == "Fake":
                    overall_risk = "MEDIUM"
                    risk_color = "#D97706"  # Amber
                    risk_message = "Either image or text appears to be fake"
                
                # Incorporate account analysis if available
                if st.session_state.get('account_analyzed', False):
                    acc_result = st.session_state.account_result
                    if acc_result[0] == "Suspicious":
                        if overall_risk == "MEDIUM":
                            overall_risk = "HIGH"
                            risk_color = "#EF4444"
                            risk_message = "Suspicious account with questionable content"
                        elif overall_risk == "LOW":
                            overall_risk = "MEDIUM"
                            risk_color = "#D97706"
                            risk_message = "Authentic content from a suspicious account"
                    elif acc_result[0] == "Attention" and overall_risk == "LOW":
                        overall_risk = "LOW-MEDIUM"
                        risk_color = "#FBBF24"
                        risk_message = "Generally authentic but with some concerns"
                
                st.markdown(f"""
                <div style="font-size: 1.8rem; color: {risk_color}; font-weight: bold; text-align: center; padding: 1rem; border-radius: 0.5rem; background-color: rgba({','.join(str(int(i)) for i in (int(risk_color[1:3], 16), int(risk_color[3:5], 16), int(risk_color[5:7], 16)))}, 0.2);">
                    {overall_risk} RISK: {risk_message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("No combined analysis performed yet.")
        
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

def display_account_analysis():
    """Display the account details analysis tab content with enhanced metrics."""
    st.markdown('<h2 class="sub-header">Account Details Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0;">Social Media Account Analysis</h3>
        <p>Enter account metrics to analyze the authenticity based on account characteristics, engagement patterns, and behavioral indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different categories of metrics
    account_tabs = st.tabs(["Basic Metrics", "Engagement Metrics", "Content Patterns", "Behavioral Indicators"])
    
    with account_tabs[0]:
        # Basic account metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Account age in days
            account_age = st.number_input("Account Age (days)", min_value=0, value=30, step=1)
            
            # Follower count
            follower_count = st.number_input("Follower Count", min_value=0, value=500, step=10)
            
            # Following count
            following_count = st.number_input("Following Count", min_value=0, value=350, step=10)
            
            # Posts count
            post_count = st.number_input("Number of Posts", min_value=0, value=25, step=1)
            
            # Account creation to first post (days)
            first_post_delay = st.number_input("Days Before First Post", min_value=0, value=1, step=1,
                                            help="Number of days between account creation and first post")
        
        with col2:
            # Profile completeness
            profile_completeness = st.slider("Profile Completeness (%)", 0, 100, 75, 5,
                                           help="How complete the profile information is (bio, picture, links, etc.)")
            
            # Account verification status
            is_verified = st.checkbox("Account is Verified", value=False)
            
            # Username characteristics
            username_type = st.radio("Username Type", 
                                    ["Natural name/word", "With numbers", "Random characters", "With special symbols"],
                                    index=0,
                                    help="The characteristic pattern of the username")
            
            # Profile has picture
            has_profile_pic = st.checkbox("Has Profile Picture", value=True)
            
            # Growth rate - followers gained per month
            follower_growth_rate = st.number_input("Monthly Follower Growth", min_value=0, value=50, step=10,
                                                help="Average number of new followers per month")
    
    with account_tabs[1]:
        # Engagement metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Like count - average per post
            avg_likes = st.number_input("Average Likes Per Post", min_value=0, value=45, step=5)
            
            # Comment count - average per post
            avg_comments = st.number_input("Average Comments Per Post", min_value=0, value=8, step=1)
            
            # Share/Retweet count - average per post
            avg_shares = st.number_input("Average Shares/Retweets Per Post", min_value=0, value=5, step=1)
            
            # Engagement rate (calculated or provided)
            engagement_rate = st.number_input("Engagement Rate (%)", min_value=0.0, value=3.5, step=0.1, format="%.1f",
                                           help="(Likes + Comments + Shares) / Followers * 100")
            
            # Engagement-to-follower ratio
            if avg_likes > 0 and follower_count > 0:
                like_follower_ratio = (avg_likes / follower_count) * 100
            else:
                like_follower_ratio = 0
            
            st.metric("Like to Follower Ratio (%)", f"{like_follower_ratio:.2f}%", 
                    help="Average likes as percentage of total followers")
        
        with col2:
            # Response rate to comments
            response_rate = st.slider("Response Rate to Comments (%)", 0, 100, 40, 5,
                                    help="How often the account responds to comments on their posts")
            
            # Average response time (hours)
            response_time = st.number_input("Average Response Time (hours)", min_value=0.0, value=8.0, step=0.5, format="%.1f",
                                         help="How quickly the account typically responds to comments")
            
            # Like distribution
            like_variability = st.slider("Like Count Consistency", 1, 10, 7, 1,
                                      help="1 = Highly variable, 10 = Very consistent")
            
            # Comment sentiment
            comment_sentiment = st.select_slider("Comment Sentiment", 
                                              options=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"],
                                              value="Positive",
                                              help="Overall sentiment of comments on posts")
            
            # Follower quality score
            follower_quality = st.slider("Follower Quality Score", 1, 10, 6, 1,
                                      help="1 = Many suspicious followers, 10 = Mostly authentic followers")
    
    with account_tabs[2]:
        # Content patterns
        col1, col2 = st.columns(2)
        
        with col1:
            # Content type distribution
            st.subheader("Content Type Distribution (%)")
            photo_percent = st.slider("Photos", 0, 100, 60, 5)
            video_percent = st.slider("Videos", 0, 100, 20, 5)
            text_percent = st.slider("Text-only", 0, 100, 20, 5)
            
            # Posting frequency (posts per week)
            posting_frequency = st.number_input("Posting Frequency (per week)", min_value=0.0, value=3.5, step=0.5, format="%.1f")
            
            # Posting time consistency
            posting_time_consistency = st.slider("Posting Time Consistency", 1, 10, 6, 1,
                                               help="1 = Random times, 10 = Consistent schedule")
        
        with col2:
            # Hashtag usage
            avg_hashtags = st.number_input("Average Hashtags Per Post", min_value=0, value=5, step=1)
            
            # Mention frequency
            avg_mentions = st.number_input("Average @Mentions Per Post", min_value=0, value=2, step=1)
            
            # Content consistency (1-10)
            content_consistency = st.slider("Content Theme Consistency", 1, 10, 8, 1,
                                          help="How consistent the account's content theme is")
            
            # Caption length
            avg_caption_length = st.number_input("Average Caption Length (words)", min_value=0, value=35, step=5)
            
            # Language consistency
            language_consistency = st.slider("Language Pattern Consistency", 1, 10, 7, 1,
                                           help="How consistent the writing style is across posts")
    
    with account_tabs[3]:
        # Behavioral indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # Geographic consistency
            geo_consistency = st.slider("Geographic Location Consistency", 1, 10, 7, 1,
                                      help="How consistent are location tags/mentions")
            
            # Device consistency
            device_consistency = st.slider("Posting Device Consistency", 1, 10, 8, 1,
                                         help="Whether posts come from the same devices/platforms")
            
            # Activity time patterns
            activity_pattern = st.select_slider("Activity Time Pattern", 
                                             options=["Very Random", "Somewhat Random", "Mixed", "Somewhat Regular", "Very Regular"],
                                             value="Somewhat Regular",
                                             help="Consistency of posting time patterns")
            
            # Has external links
            has_external_links = st.checkbox("Has External Links in Bio", value=True)
        
        with col2:
            # Bio-content consistency
            bio_content_match = st.slider("Bio-Content Consistency", 1, 10, 8, 1,
                                        help="How well the content matches what's described in the bio")
            
            # Active days distribution
            weekday_bias = st.selectbox("Active Days Pattern", 
                                      ["Mostly weekdays", "Mostly weekends", "Evenly distributed", "Very irregular"],
                                      index=2,
                                      help="When the account is typically active")
            
            # Comment-to-post ratio
            if post_count > 0:
                comments_made = st.number_input("Comments Made on Other Accounts", min_value=0, value=120, step=10)
                comment_post_ratio = comments_made / post_count
            else:
                comments_made = st.number_input("Comments Made on Other Accounts", min_value=0, value=0, step=10)
                comment_post_ratio = 0
                
            st.metric("Comment to Post Ratio", f"{comment_post_ratio:.1f}",
                    help="Ratio of comments made vs posts created")
            
            # Last activity (days ago)
            last_activity = st.number_input("Days Since Last Activity", min_value=0, value=2, step=1,
                                          help="Number of days since the last post or comment")
    
    # Analyze button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("Analyze Account", key="analyze_account_btn", use_container_width=True)
    
    # Process the analysis when button clicked
    if analyze_clicked:
        with st.spinner("Analyzing account metrics..."):
            # Add artificial delay for better UX
            time.sleep(1.5)
            
            # Get the analysis result with enhanced metrics
            result, metrics = analyze_enhanced_account_metrics(
                # Basic metrics
                account_age, follower_count, following_count, post_count, profile_completeness,
                is_verified, username_type, has_profile_pic, follower_growth_rate, first_post_delay,
                
                # Engagement metrics
                avg_likes, avg_comments, avg_shares, engagement_rate, response_rate,
                response_time, like_variability, comment_sentiment, follower_quality,
                
                # Content patterns
                posting_frequency, content_consistency, posting_time_consistency,
                avg_hashtags, avg_mentions, language_consistency, avg_caption_length,
                {
                    "photos": photo_percent,
                    "videos": video_percent,
                    "text": text_percent
                },
                
                # Behavioral indicators
                geo_consistency, device_consistency, activity_pattern, has_external_links,
                bio_content_match, weekday_bias, comment_post_ratio, last_activity
            )
            
            # Update session state
            st.session_state.account_analyzed = True
            st.session_state.account_result = (result, metrics)
            st.session_state.analysis_count += 1
            st.session_state.total_accounts = st.session_state.get('total_accounts', 0) + 1
            st.session_state.total_risk_score = st.session_state.get('total_risk_score', 0) + metrics['risk_score']
            
            if result == "Suspicious":
                st.session_state.fake_detections += 1
            
            # Display the result with appropriate styling
            st.markdown("<h3 class='sub-header'>Analysis Result:</h3>", unsafe_allow_html=True)
            
            if result == "Authentic":
                st.markdown(f"<div class='result-real'>✅ LIKELY AUTHENTIC ACCOUNT</div>", unsafe_allow_html=True)
            elif result == "Attention":
                st.markdown(f"<div class='result-fake' style='background-color: #FEF3C7; color: #D97706;'>⚠️ REQUIRES ATTENTION</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-fake'>⚠️ SUSPICIOUS ACCOUNT DETECTED</div>", unsafe_allow_html=True)
            
            st.markdown(f"<div class='confidence'>Confidence: {metrics['confidence']:.2f}%</div>", unsafe_allow_html=True)
            
            # Display risk score
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Create risk gauge
            risk_score = metrics['risk_score']
            gauge_colors = ['#10B981', '#FBBF24', '#EF4444']
            bar_height = 0.6
            
            # Draw the gauge bar background
            ax.barh(0, 100, height=bar_height, color='#e5e7eb')
            
            # Draw the gauge value with appropriate color
            if risk_score <= 33:
                color = gauge_colors[0]  # Green for low risk
            elif risk_score <= 66:
                color = gauge_colors[1]  # Yellow for medium risk
            else:
                color = gauge_colors[2]  # Red for high risk
            
            ax.barh(0, risk_score, height=bar_height, color=color)
            
            # Add marker for threshold
            ax.axvline(x=33, color='black', linestyle='--', alpha=0.7)
            ax.axvline(x=66, color='black', linestyle='--', alpha=0.7)
            
            # Add labels
            ax.text(15, -0.8, "Low Risk", fontsize=10)
            ax.text(45, -0.8, "Medium Risk", fontsize=10)
            ax.text(80, -0.8, "High Risk", fontsize=10)
            
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
            
            # Display metrics summary with tabs for different categories
            st.markdown("<h3 class='sub-header'>Detailed Analysis</h3>", unsafe_allow_html=True)
            
            # Create tabs for different categories of analysis
            analysis_tabs = st.tabs(["Overview", "Risk Factors", "Authenticity Signals", "Recommendations"])
            
            with analysis_tabs[0]:
                # Display summary metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    # Account summary
                    st.markdown("<h4>Account Summary</h4>", unsafe_allow_html=True)
                    
                    account_type = "Personal" if follower_count < 10000 else "Influencer" if follower_count < 100000 else "Brand/Celebrity"
                    account_activity = "Very Active" if posting_frequency > 5 else "Active" if posting_frequency > 2 else "Moderate" if posting_frequency > 0.5 else "Low Activity"
                    engagement_quality = "Very High" if engagement_rate > 8 else "High" if engagement_rate > 4 else "Moderate" if engagement_rate > 2 else "Low"
                    
                    st.markdown(f"**Account Type:** {account_type}")
                    st.markdown(f"**Account Age:** {account_age} days")
                    st.markdown(f"**Activity Level:** {account_activity}")
                    st.markdown(f"**Engagement Quality:** {engagement_quality}")
                    st.markdown(f"**Overall Risk Score:** {risk_score:.1f}/100")
                
                with col2:
                    # Visual summary
                    st.markdown("<h4>Key Metrics</h4>", unsafe_allow_html=True)
                    
                    # Create a radar chart for key metrics
                    categories = ['Followers', 'Engagement', 'Content\nConsistency', 'Behavior\nPatterns', 'Account\nActivity']
                    
                    # Normalize values to 0-1 scale
                    followers_score = min(1.0, follower_count / 10000)  # Scale up to 10K
                    engagement_score = min(1.0, engagement_rate / 10)   # Scale up to 10%
                    consistency_score = content_consistency / 10
                    behavior_score = (geo_consistency + device_consistency) / 20
                    activity_score = min(1.0, posting_frequency / 7)    # Scale up to 7 posts/week
                    
                    values = [followers_score, engagement_score, consistency_score, behavior_score, activity_score]
                    
                    # Create radar chart
                    fig = plt.figure(figsize=(4, 4))
                    ax = fig.add_subplot(111, polar=True)
                    
                    # Plot data
                    values_with_closure = values + [values[0]]
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                    angles_with_closure = angles + [angles[0]]
                    
                    ax.plot(angles_with_closure, values_with_closure, 'o-', linewidth=2)
                    ax.fill(angles_with_closure, values_with_closure, alpha=0.25)
                    ax.set_thetagrids(np.degrees(angles), categories)
                    ax.set_ylim(0, 1)
                    ax.grid(True)
                    
                    st.pyplot(fig)
            
            with analysis_tabs[1]:
                # Risk factors
                st.markdown("<h4>Identified Risk Factors</h4>", unsafe_allow_html=True)
                
                if metrics['suspicious_indicators']:
                    # Create columns for different categories of risk
                    col1, col2 = st.columns(2)
                    
                    # Split the indicators into two columns
                    indicators = list(metrics['suspicious_indicators'].items())
                    half = len(indicators) // 2
                    
                    with col1:
                        for key, value in indicators[:half]:
                            st.markdown(f"• **{key}:** {value}")
                    
                    with col2:
                        for key, value in indicators[half:]:
                            st.markdown(f"• **{key}:** {value}")
                    
                    # Add detailed impact explanation
                    with st.expander("Impact of Risk Factors", expanded=False):
                        st.markdown("""
                        Risk factors are patterns or characteristics that correlate with inauthentic accounts. Each factor 
                        contributes to the overall risk score, with more severe factors having higher impact. The presence of
                        multiple related risk factors indicates a higher probability of a suspicious account.
                        """)
                        
                        # Show top 3 factors with detailed explanation
                        st.markdown("### Top Impact Factors")
                        for i, (key, value) in enumerate(sorted(metrics['suspicious_indicators'].items(), 
                                                           key=lambda x: metrics.get('factor_weights', {}).get(x[0], 0), 
                                                           reverse=True)[:3]):
                            st.markdown(f"**{i+1}. {key}:** {value}")
                            factor_explanation = {
                                "Low Engagement": "Extremely low engagement relative to follower count is a strong indicator of purchased followers or bot activity.",
                                "New Account": "Recently created accounts with high follower counts often indicate inauthentic growth.",
                                "Imbalanced Following Ratio": "Accounts following many users with few followers suggests follow-for-follow tactics.",
                                "Few Posts, Many Followers": "High follower counts with minimal content is unusual for organic growth.",
                                "Inconsistent Content": "Rapid shifts in content type or subject can indicate purchased accounts.",
                                "Geographic Inconsistency": "Posts from widely different locations in short timeframes is physically impossible.",
                                "Excessive Posting": "Extremely high posting frequency often indicates automation.",
                                "Incomplete Profile": "Authentic accounts typically complete their profiles.",
                                "Random Username": "Usernames with random characters or excessive numbers correlate with bot accounts.",
                                "Device Inconsistency": "Multiple different posting devices may indicate a shared account."
                            }.get(key, "This pattern is commonly associated with inauthentic accounts.")
                            
                            st.markdown(f"*{factor_explanation}*")
                else:
                    st.markdown("No significant risk factors identified.")
            
            with analysis_tabs[2]:
                # Authenticity signals
                st.markdown("<h4>Authenticity Indicators</h4>", unsafe_allow_html=True)
                
                if metrics['positive_indicators']:
                    # Create columns for different categories of positive indicators
                    col1, col2 = st.columns(2)
                    
                    # Split the indicators into two columns
                    indicators = list(metrics['positive_indicators'].items())
                    half = len(indicators) // 2
                    
                    with col1:
                        for key, value in indicators[:half]:
                            st.markdown(f"• **{key}:** {value}")
                    
                    with col2:
                        for key, value in indicators[half:]:
                            st.markdown(f"• **{key}:** {value}")
                    
                    # Add authenticity score gauge
                    authenticity_score = 100 - risk_score
                    
                    # Create gauge chart
                    fig, ax = plt.subplots(figsize=(8, 1))
                    gauge_colors = ['#EF4444', '#FBBF24', '#10B981']
                    bar_height = 0.4
                    
                    # Draw the gauge bar background
                    ax.barh(0, 100, height=bar_height, color='#e5e7eb')
                    
                    # Draw the gauge value
                    if authenticity_score < 33:
                        color = gauge_colors[0]  # Red for low authenticity
                    elif authenticity_score < 66:
                        color = gauge_colors[1]  # Yellow for medium authenticity
                    else:
                        color = gauge_colors[2]  # Green for high authenticity
                    
                    ax.barh(0, authenticity_score, height=bar_height, color=color)
                    
                    # Add marker for threshold
                    ax.axvline(x=33, color='black', linestyle='--', alpha=0.7)
                    ax.axvline(x=66, color='black', linestyle='--', alpha=0.7)
                    
                    # Add labels
                    ax.text(10, -0.8, "Low Authenticity", fontsize=9)
                    ax.text(40, -0.8, "Medium", fontsize=9)
                    ax.text(80, -0.8, "High Authenticity", fontsize=9)
                    ax.text(authenticity_score - 5, 0, f"{authenticity_score:.0f}", fontsize=10, va='center', ha='right')
                    
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
                    
                    # Engagement quality assessment
                    st.markdown("#### Engagement Quality Assessment")
                    
                    engagement_quality_score = min(10, max(1, int(10 * (
                        (min(1, engagement_rate/10) * 0.4) + 
                        (min(1, response_rate/100) * 0.3) + 
                        (min(1, follower_quality/10) * 0.3)
                    ))))
                    
                    engagement_descriptions = {
                        1: "Very poor engagement with potential fake interactions",
                        2: "Poor engagement suggesting low quality followers",
                        3: "Below average engagement with limited interaction",
                        4: "Fair engagement but inconsistent interaction patterns",
                        5: "Average engagement typical of authentic accounts",
                        6: "Good engagement with active community",
                        7: "Strong engagement showing authentic followers",
                        8: "Very strong engagement with high-quality interactions",
                        9: "Excellent engagement demonstrating loyal following",
                        10: "Exceptional engagement typical of highly authentic accounts"
                    }
                    
                    st.markdown(f"**Engagement Quality Score: {engagement_quality_score}/10** - *{engagement_descriptions[engagement_quality_score]}*")
                else:
                    st.markdown("No significant authenticity indicators identified.")
            
            with analysis_tabs[3]:
                # Recommendations
                st.markdown("<h4>Actionable Recommendations</h4>", unsafe_allow_html=True)
                
                for rec in metrics['recommendations']:
                    st.markdown(f"• {rec}")
                
                # Display comparison to benchmarks if we have any
                st.markdown("#### Comparison to Industry Benchmarks")
                
                # Create a comparison table
                benchmark_data = {
                    "Metric": ["Engagement Rate", "Posting Frequency", "Follow Ratio", "Follower Growth"],
                    "Account": [f"{engagement_rate:.1f}%", f"{posting_frequency:.1f}/week", 
                              f"{follower_ratio:.2f}", f"{follower_growth_rate} per month"],
                    "Industry Average": ["2.4%", "4.5/week", "0.85", "3% monthly"]
                }
                
                benchmark_df = pd.DataFrame(benchmark_data)
                st.table(benchmark_df)
                
                # Add verification steps
                with st.expander("Verification Steps", expanded=False):
                    st.markdown("""
                    ### Recommended Verification Steps
                    
                    1. **Check content history**: Review older posts for consistency with current content
                    2. **Examine follower profiles**: Look for suspicious patterns among followers
                    3. **Verify through alternate channels**: Check for presence on other platforms
                    4. **Look for engagement patterns**: Authentic accounts have varied engagement
                    5. **Analyze comment quality**: Check if comments are generic or specific to content
                    """)

def analyze_enhanced_account_metrics(
    # Basic metrics
    account_age, follower_count, following_count, post_count, profile_completeness,
    is_verified, username_type, has_profile_pic, follower_growth_rate, first_post_delay,
    
    # Engagement metrics
    avg_likes, avg_comments, avg_shares, engagement_rate, response_rate,
    response_time, like_variability, comment_sentiment, follower_quality,
    
    # Content patterns
    posting_frequency, content_consistency, posting_time_consistency,
    avg_hashtags, avg_mentions, language_consistency, avg_caption_length,
    content_type_distribution,
    
    # Behavioral indicators
    geo_consistency, device_consistency, activity_pattern, has_external_links,
    bio_content_match, weekday_bias, comment_post_ratio, last_activity
):
    """
    Analyze comprehensive social media account metrics to determine authenticity.
    
    Returns:
        tuple: (prediction, metrics_dict)
    """
    # Initialize metrics dictionary
    metrics = {
        'confidence': 0.0,
        'risk_score': 0.0,
        'positive_indicators': {},
        'suspicious_indicators': {},
        'recommendations': [],
        'factor_weights': {}  # Store weights of factors for explanation
    }
    
    # Calculate follower-to-following ratio (if applicable)
    follower_ratio = follower_count / max(following_count, 1)
    
    # Initialize risk score (0-100, higher means more suspicious)
    risk_score = 50  # Start at neutral
    
    # --- Analyze account age and creation patterns ---
    if account_age < 30:
        weight = 15
        risk_score += weight
        metrics['suspicious_indicators']['New Account'] = f"{account_age} days old"
        metrics['recommendations'].append("New accounts with high follower counts warrant additional verification")
        metrics['factor_weights']['New Account'] = weight
    else:
        weight = 10
        risk_score -= weight
        metrics['positive_indicators']['Established Account'] = f"{account_age} days old"
        metrics['factor_weights']['Established Account'] = weight
    
    if first_post_delay > 7 and follower_count > 1000:
        weight = 8
        risk_score += weight
        metrics['suspicious_indicators']['Delayed First Post'] = f"{first_post_delay} days after creation"
        metrics['factor_weights']['Delayed First Post'] = weight
    
    # --- Analyze username patterns ---
    if username_type in ["Random characters", "With special symbols"]:
        weight = 10
        risk_score += weight
        metrics['suspicious_indicators']['Suspicious Username Pattern'] = username_type
        metrics['factor_weights']['Suspicious Username Pattern'] = weight
    elif username_type == "Natural name/word":
        weight = 5
        risk_score -= weight
        metrics['positive_indicators']['Natural Username'] = username_type
        metrics['factor_weights']['Natural Username'] = weight
    
    # --- Analyze follower/following patterns ---
    if follower_count > 1000 and follower_ratio < 0.1:
        weight = 15
        risk_score += weight
        metrics['suspicious_indicators']['Imbalanced Following Ratio'] = f"{follower_ratio:.2f}"
        metrics['recommendations'].append("Accounts following many users with few followers often use follow/unfollow tactics")
        metrics['factor_weights']['Imbalanced Following Ratio'] = weight
    elif follower_ratio > 0.5:
        weight = 10
        risk_score -= weight
        metrics['positive_indicators']['Healthy Following Ratio'] = f"{follower_ratio:.2f}"
        metrics['factor_weights']['Healthy Following Ratio'] = weight
    
    # Analyze follower growth rate
    monthly_growth_percent = (follower_growth_rate / max(follower_count, 1)) * 100
    if monthly_growth_percent > 30 and account_age < 90:
        weight = 12
        risk_score += weight
        metrics['suspicious_indicators']['Unusual Growth Rate'] = f"{monthly_growth_percent:.1f}% monthly"
        metrics['recommendations'].append("Extremely rapid follower growth may indicate purchased followers")
        metrics['factor_weights']['Unusual Growth Rate'] = weight
    elif monthly_growth_percent < 15 and monthly_growth_percent > 1:
        weight = 8
        risk_score -= weight
        metrics['positive_indicators']['Natural Growth Rate'] = f"{monthly_growth_percent:.1f}% monthly"
        metrics['factor_weights']['Natural Growth Rate'] = weight
    
    # --- Analyze engagement metrics ---
    expected_likes = follower_count * 0.03  # 3% is a reasonable baseline
    like_deviation = abs(avg_likes - expected_likes) / max(expected_likes, 1)
    
    if follower_count > 1000 and avg_likes < (follower_count * 0.005):  # Less than 0.5% engagement
        weight = 20
        risk_score += weight
        metrics['suspicious_indicators']['Low Engagement'] = f"{avg_likes} likes vs {follower_count} followers"
        metrics['recommendations'].append("Very low engagement despite high follower count suggests purchased followers")
        metrics['factor_weights']['Low Engagement'] = weight
    elif like_deviation < 0.5 and follower_count > 100:
        weight = 15
        risk_score -= weight
        metrics['positive_indicators']['Natural Engagement'] = f"{(avg_likes/follower_count*100):.1f}% engagement rate"
        metrics['factor_weights']['Natural Engagement'] = weight
    
    # Analyze comment patterns
    if avg_comments > 0 and comment_sentiment in ["Positive", "Very Positive"]:
        weight = 8
        risk_score -= weight
        metrics['positive_indicators']['Positive Comment Sentiment'] = comment_sentiment
        metrics['factor_weights']['Positive Comment Sentiment'] = weight
    
    # Analyze response rates
    if response_rate > 50:
        weight = 10
        risk_score -= weight
        metrics['positive_indicators']['Active Community Engagement'] = f"{response_rate}% response rate"
        metrics['factor_weights']['Active Community Engagement'] = weight
    elif response_rate < 10 and follower_count > 1000:
        weight = 8
        risk_score += weight
        metrics['suspicious_indicators']['Low Audience Interaction'] = f"{response_rate}% response rate"
        metrics['factor_weights']['Low Audience Interaction'] = weight
    
    # --- Analyze content patterns ---
    if post_count < 5 and follower_count > 1000:
        weight = 20
        risk_score += weight
        metrics['suspicious_indicators']['Few Posts, Many Followers'] = f"{post_count} posts, {follower_count} followers"
        metrics['recommendations'].append("High follower count with minimal content is a strong indicator of inauthenticity")
        metrics['factor_weights']['Few Posts, Many Followers'] = weight
    
    # Analyze posting frequency
    if posting_frequency > 0 and posting_frequency < 20:
        weight = 5
        risk_score -= weight
        metrics['positive_indicators']['Regular Posting Schedule'] = f"{posting_frequency} posts/week"
        metrics['factor_weights']['Regular Posting Schedule'] = weight
    elif posting_frequency > 50:
        weight = 10
        risk_score += weight
        metrics['suspicious_indicators']['Excessive Posting'] = f"{posting_frequency} posts/week"
        metrics['recommendations'].append("Extremely high posting frequency can indicate automation")
        metrics['factor_weights']['Excessive Posting'] = weight
    
    # Analyze content consistency
    if content_consistency < 3:
        weight = 12
        risk_score += weight
        metrics['suspicious_indicators']['Inconsistent Content'] = f"{content_consistency}/10"
        metrics['recommendations'].append("Inconsistent content themes may indicate purchased or repurposed accounts")
        metrics['factor_weights']['Inconsistent Content'] = weight
    elif content_consistency > 7:
        weight = 10
        risk_score -= weight
        metrics['positive_indicators']['Consistent Content'] = f"{content_consistency}/10"
        metrics['factor_weights']['Consistent Content'] = weight
    
    # Analyze hashtag usage
    if avg_hashtags > 20:
        weight = 5
        risk_score += weight
        metrics['suspicious_indicators']['Excessive Hashtags'] = f"{avg_hashtags} per post"
        metrics['factor_weights']['Excessive Hashtags'] = weight
    
    # Analyze language consistency
    if language_consistency > 8:
        weight = 8
        risk_score -= weight
        metrics['positive_indicators']['Consistent Voice'] = f"{language_consistency}/10"
        metrics['factor_weights']['Consistent Voice'] = weight
    elif language_consistency < 4:
        weight = 6
        risk_score += weight
        metrics['suspicious_indicators']['Inconsistent Language'] = f"{language_consistency}/10"
        metrics['recommendations'].append("Inconsistent writing style may indicate multiple authors or AI-generated content")
        metrics['factor_weights']['Inconsistent Language'] = weight
    
    # --- Analyze profile completeness and verification ---
    if profile_completeness < 30:
        weight = 10
        risk_score += weight
        metrics['suspicious_indicators']['Incomplete Profile'] = f"{profile_completeness}% complete"
        metrics['recommendations'].append("Complete profiles increase perceived authenticity")
        metrics['factor_weights']['Incomplete Profile'] = weight
    elif profile_completeness > 80:
        weight = 10
        risk_score -= weight
        metrics['positive_indicators']['Complete Profile'] = f"{profile_completeness}% complete"
        metrics['factor_weights']['Complete Profile'] = weight
    
    if is_verified:
        weight = 25
        risk_score -= weight
        metrics['positive_indicators']['Verified Account'] = "Yes"
        metrics['factor_weights']['Verified Account'] = weight
    
    if not has_profile_pic:
        weight = 15
        risk_score += weight
        metrics['suspicious_indicators']['No Profile Picture'] = "Missing profile image"
        metrics['factor_weights']['No Profile Picture'] = weight
    
    # --- Analyze behavioral patterns ---
    if geo_consistency < 3:
        weight = 12
        risk_score += weight
        metrics['suspicious_indicators']['Geographic Inconsistency'] = f"{geo_consistency}/10"
        metrics['recommendations'].append("Inconsistent geographic patterns may indicate shared accounts or location spoofing")
        metrics['factor_weights']['Geographic Inconsistency'] = weight
    elif geo_consistency > 7:
        weight = 5
        risk_score -= weight
        metrics['positive_indicators']['Geographic Consistency'] = f"{geo_consistency}/10"
        metrics['factor_weights']['Geographic Consistency'] = weight
    
    if device_consistency < 4:
        weight = 8
        risk_score += weight
        metrics['suspicious_indicators']['Device Inconsistency'] = f"{device_consistency}/10"
        metrics['factor_weights']['Device Inconsistency'] = weight
    elif device_consistency > 8:
        weight = 5
        risk_score -= weight
        metrics['positive_indicators']['Device Consistency'] = f"{device_consistency}/10"
        metrics['factor_weights']['Device Consistency'] = weight
    
    if activity_pattern in ["Very Random", "Somewhat Random"] and posting_frequency > 10:
        weight = 10
        risk_score += weight
        metrics['suspicious_indicators']['Erratic Posting Pattern'] = activity_pattern
        metrics['recommendations'].append("Highly irregular posting times with high frequency suggests automation")
        metrics['factor_weights']['Erratic Posting Pattern'] = weight
    elif activity_pattern in ["Somewhat Regular", "Very Regular"]:
        weight = 5
        risk_score -= weight
        metrics['positive_indicators']['Consistent Activity Pattern'] = activity_pattern
        metrics['factor_weights']['Consistent Activity Pattern'] = weight
    
    # Analyze bio-content match
    if bio_content_match < 4:
        weight = 8
        risk_score += weight
        metrics['suspicious_indicators']['Bio-Content Mismatch'] = f"{bio_content_match}/10"
        metrics['factor_weights']['Bio-Content Mismatch'] = weight
    elif bio_content_match > 7:
        weight = 7
        risk_score -= weight
        metrics['positive_indicators']['Bio Matches Content'] = f"{bio_content_match}/10"
        metrics['factor_weights']['Bio Matches Content'] = weight
    
    # --- Analyze external links ---
    if has_external_links and account_age < 30 and follower_count > 5000:
        weight = 10
        risk_score += weight
        metrics['suspicious_indicators']['New Account with Links'] = "Suspicious pattern"
        metrics['recommendations'].append("New accounts with links and many followers often indicate promotional accounts")
        metrics['factor_weights']['New Account with Links'] = weight
    elif has_external_links and account_age > 180:
        weight = 5
        risk_score -= weight
        metrics['positive_indicators']['Established with Web Presence'] = "Yes"
        metrics['factor_weights']['Established with Web Presence'] = weight
    
    # --- Analyze follower quality ---
    if follower_quality < 4 and follower_count > 1000:
        weight = 15
        risk_score += weight
        metrics['suspicious_indicators']['Low Quality Followers'] = f"{follower_quality}/10"
        metrics['recommendations'].append("A high percentage of suspicious followers suggests purchased followers")
        metrics['factor_weights']['Low Quality Followers'] = weight
    elif follower_quality > 7:
        weight = 12
        risk_score -= weight
        metrics['positive_indicators']['High Quality Followers'] = f"{follower_quality}/10"
        metrics['factor_weights']['High Quality Followers'] = weight
    
    # --- Analyze comment-to-post ratio ---
    if comment_post_ratio > 5:
        weight = 8
        risk_score -= weight
        metrics['positive_indicators']['Active Community Member'] = f"{comment_post_ratio:.1f} comments per post"
        metrics['factor_weights']['Active Community Member'] = weight
    elif comment_post_ratio < 0.5 and post_count > 20:
        weight = 5
        risk_score += weight
        metrics['suspicious_indicators']['Low Community Participation'] = f"{comment_post_ratio:.1f} comments per post"
        metrics['factor_weights']['Low Community Participation'] = weight
    
    # --- Analyze recency of activity ---
    if last_activity > 30 and follower_growth_rate > 0:
        weight = 10
        risk_score += weight
        metrics['suspicious_indicators']['Growing But Inactive'] = f"No activity for {last_activity} days"
        metrics['recommendations'].append("Growing follower count despite account inactivity is suspicious")
        metrics['factor_weights']['Growing But Inactive'] = weight
    
    # Ensure risk score stays within 0-100 range
    risk_score = max(0, min(100, risk_score))
    metrics['risk_score'] = risk_score
    
    # Determine prediction based on risk score
    if risk_score < 40:
        prediction = "Authentic"
        confidence = 100 - risk_score
    elif risk_score < 70:
        prediction = "Attention"
        confidence = 100 - abs(risk_score - 50)
    else:
        prediction = "Suspicious"
        confidence = risk_score
    
    metrics['confidence'] = confidence
    
    # Add general recommendations based on account type
    if len(metrics['recommendations']) == 0:
        if follower_count > 10000:
            metrics['recommendations'].append("Verify this influencer account through other social media platforms")
            metrics['recommendations'].append("Check for verified badges on other platforms")
        else:
            metrics['recommendations'].append("Continue monitoring account behavior for changes")
            metrics['recommendations'].append("Verify account through alternate communication channels if needed")
    
    # Add specific verification recommendations
    if "Low Engagement" in metrics['suspicious_indicators']:
        metrics['recommendations'].append("Analyze a sample of followers to check for bot-like profiles")
    
    if "Inconsistent Content" in metrics['suspicious_indicators']:
        metrics['recommendations'].append("Review older posts to check for sudden changes in content style or quality")
    
    return prediction, metrics

def display_combined_analysis():
    """Display the combined analysis tab content."""
    st.markdown('<h2 class="sub-header">Combined Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3 style="margin-top: 0;">Comprehensive Content Analysis</h3>
        <p>Upload an image, enter text content, and provide account details for a complete authenticity assessment.</p>
        <p>This combined approach provides a more thorough evaluation by examining multiple aspects of social media content.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different inputs
    input_tabs = st.tabs(["Image", "Text", "Account Details"])
    
    with input_tabs[0]:
        # Load the model
        model = load_detection_model()
        
        if model is None:
            st.warning("Please make sure the model file 'my_model.keras' is in the same directory as this script.")
        else:
            # File uploader
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="combined_image_upload")
            
            # Display the uploaded image
            if uploaded_file is not None:
                try:
                    # Display the uploaded image
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Image", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error displaying the image: {e}")
                    uploaded_file = None
    
    with input_tabs[1]:
        # Model selection
        model_options = {"Logistic Regression": "logistic", "Random Forest": "random_forest"}
        selected_model_name = st.selectbox("Select Text Analysis Model", list(model_options.keys()), key="combined_text_model")
        model_type = model_options[selected_model_name]
        
        # Text input
        text_input = st.text_area(
            "Paste social media text for analysis",
            value="",
            height=150,
            key="combined_text_input"
        )
    
    with input_tabs[2]:
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Account age in days/months
            account_age = st.number_input("Account Age (days)", min_value=0, value=0, step=1, key="combined_account_age")
            
            # Follower count
            follower_count = st.number_input("Follower Count", min_value=0, value=0, step=1, key="combined_follower_count")
            
            # Following count
            following_count = st.number_input("Following Count", min_value=0, value=0, step=1, key="combined_following_count")
            
            # Posts count
            post_count = st.number_input("Number of Posts", min_value=0, value=0, step=1, key="combined_post_count")
        
        with col2:
            # Engagement rate (average likes/comments per post)
            engagement_rate = st.number_input("Engagement Rate (%)", min_value=0.0, value=0.0, step=0.1, format="%.1f", key="combined_engagement_rate")
            
            # Profile completeness
            profile_completeness = st.slider("Profile Completeness", 0, 100, 50, 5, key="combined_profile_completeness")
            
            # Posting frequency (posts per week)
            posting_frequency = st.number_input("Posting Frequency (per week)", min_value=0.0, value=0.0, step=0.1, format="%.1f", key="combined_posting_frequency")
            
            # Account verification status
            is_verified = st.checkbox("Account is Verified", key="combined_is_verified")
        
        # Additional parameters expandable section
        with st.expander("Additional Parameters"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Content consistency (1-10)
                content_consistency = st.slider("Content Consistency", 1, 10, 5, 1, key="combined_content_consistency",
                                              help="How consistent is the account's content theme and quality")
            
            with col2:
                # Link presence in bio
                has_external_links = st.checkbox("Has External Links in Bio", key="combined_has_external_links")
                
                # Geographic consistency
                geo_consistency = st.slider("Geographic Consistency", 1, 10, 5, 1, key="combined_geo_consistency",
                                          help="How consistent are location tags/mentions")
    
    # Analyze button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_clicked = st.button("Analyze All Content", key="analyze_combined_all_btn", use_container_width=True)
    
    # Process the analysis when button clicked
    if analyze_clicked:
        missing_inputs = []
        
        if uploaded_file is None:
            missing_inputs.append("image")
        
        if not text_input:
            missing_inputs.append("text content")
        
        if follower_count == 0 and following_count == 0 and post_count == 0:
            missing_inputs.append("account details")
        
        if missing_inputs:
            if len(missing_inputs) == 1:
                st.warning(f"Please provide {missing_inputs[0]} to analyze.")
            elif len(missing_inputs) == 2:
                st.warning(f"Please provide {missing_inputs[0]} and {missing_inputs[1]} to analyze.")
            else:
                st.warning("Please provide image, text content, and account details to analyze.")
        else:
            with st.spinner("Performing comprehensive analysis..."):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Process image
                img = Image.open(uploaded_file)
                progress_bar.progress(15)
                time.sleep(0.3)  # Small delay for better UX
                
                # Resize image to match model's expected input
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized)
                
                # If image has 4 channels (RGBA), convert to 3 channels (RGB)
                if img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                
                # Make image prediction
                image_prediction, image_confidence = predict_image(img_array, model)
                progress_bar.progress(35)
                time.sleep(0.3)  # Small delay for better UX
                
                # Process text
                text_prediction, text_confidence, text_explanation = predict_account_type(text_input, model_type)
                progress_bar.progress(55)
                time.sleep(0.3)  # Small delay for better UX
                
                # Process account metrics
                account_prediction, account_metrics = analyze_account_metrics(
                    account_age, 
                    follower_count, 
                    following_count, 
                    post_count, 
                    engagement_rate, 
                    profile_completeness,
                    posting_frequency,
                    is_verified,
                    content_consistency,
                    has_external_links,
                    geo_consistency
                )
                progress_bar.progress(75)
                time.sleep(0.3)  # Small delay for better UX
                
                # Update session state
                st.session_state.combined_analyzed = True
                st.session_state.image_analyzed = True
                st.session_state.text_analyzed = True
                st.session_state.account_analyzed = True
                st.session_state.image_result = (image_prediction, image_confidence)
                st.session_state.text_result = (text_prediction, text_confidence, text_explanation)
                st.session_state.account_result = (account_prediction, account_metrics)
                st.session_state.analysis_count += 1
                st.session_state.total_images += 1
                st.session_state.total_texts += 1
                st.session_state.total_accounts += 1
                st.session_state.total_risk_score = st.session_state.get('total_risk_score', 0) + account_metrics['risk_score']
                
                if image_prediction == "Fake" or text_prediction == "Fake" or account_prediction == "Suspicious":
                    st.session_state.fake_detections += 1
                
                progress_bar.progress(100)
                
            # Display results with tabs
            st.markdown('<h3 class="sub-header">Analysis Results</h3>', unsafe_allow_html=True)
            
            result_tabs = st.tabs(["Overall Assessment", "Image Analysis", "Text Analysis", "Account Analysis"])
            
            with result_tabs[0]:
                # Calculate composite risk score (weighted average)
                image_risk = 100 - image_confidence if image_prediction == "Real" else image_confidence
                text_risk = 100 - text_confidence*100 if text_prediction == "Real" else text_confidence*100
                account_risk = account_metrics['risk_score']
                
                # Weights could be adjusted based on importance
                composite_risk = (image_risk * 0.3) + (text_risk * 0.3) + (account_risk * 0.4)
                
                # Determine risk level
                if composite_risk < 33:
                    risk_level = "LOW"
                    risk_color = "#10B981"  # Green
                elif composite_risk < 66:
                    risk_level = "MEDIUM"
                    risk_color = "#D97706"  # Amber
                else:
                    risk_level = "HIGH"
                    risk_color = "#EF4444"  # Red
                
                # Generate risk message
                risk_factors = []
                if image_prediction == "Fake":
                    risk_factors.append("manipulated image")
                if text_prediction == "Fake":
                    risk_factors.append("inauthentic text content")
                if account_prediction == "Suspicious":
                    risk_factors.append("suspicious account metrics")
                
                if risk_factors:
                    if len(risk_factors) == 1:
                        risk_message = f"Analysis detected {risk_factors[0]}"
                    elif len(risk_factors) == 2:
                        risk_message = f"Analysis detected {risk_factors[0]} and {risk_factors[1]}"
                    else:
                        risk_message = f"Analysis detected {risk_factors[0]}, {risk_factors[1]}, and {risk_factors[2]}"
                else:
                    risk_message = "Content appears authentic across all analysis dimensions"
                
                # Display overall assessment
                st.markdown(f"""
                <div style="font-size: 1.8rem; color: {risk_color}; font-weight: bold; text-align: center; padding: 1rem; border-radius: 0.5rem; background-color: rgba({','.join(str(int(i)) for i in (int(risk_color[1:3], 16), int(risk_color[3:5], 16), int(risk_color[5:7], 16)))}, 0.2); margin-bottom: 1.5rem;">
                    {risk_level} RISK: {risk_message}
                </div>
                """, unsafe_allow_html=True)
                
                # Create risk gauge
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Draw the gauge bar background
                bar_height = 0.6
                ax.barh(0, 100, height=bar_height, color='#e5e7eb')
                
                # Draw the gauge value with color based on risk level
                ax.barh(0, composite_risk, height=bar_height, color=risk_color)
                
                # Add markers for thresholds
                ax.axvline(x=33, color='black', linestyle='--', alpha=0.7)
                ax.axvline(x=66, color='black', linestyle='--', alpha=0.7)
                
                # Add labels
                ax.text(15, -0.8, "Low Risk", fontsize=10)
                ax.text(45, -0.8, "Medium Risk", fontsize=10)
                ax.text(80, -0.8, "High Risk", fontsize=10)
                ax.text(composite_risk + 2, 0, f"{composite_risk:.1f}%", va='center', fontsize=11, fontweight='bold')
                
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
                
                # Display summary table
                st.markdown("<h4>Analysis Summary</h4>", unsafe_allow_html=True)
                
                # Create summary table
                summary_data = {
                    "Analysis Type": ["Image Analysis", "Text Analysis", "Account Analysis"],
                    "Result": [
                        f"{'✅ Real' if image_prediction == 'Real' else '⚠️ Deepfake'}", 
                        f"{'✅ Authentic' if text_prediction == 'Real' else '⚠️ Fake'}", 
                        f"{'✅ Authentic' if account_prediction == 'Authentic' else '⚠️ Suspicious' if account_prediction == 'Suspicious' else '⚠️ Attention Required'}"
                    ],
                    "Confidence/Risk": [
                        f"{image_confidence:.1f}%",
                        f"{text_confidence:.1%}",
                        f"{account_metrics['risk_score']:.1f}/100"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)
                
                # Display key indicators
                st.markdown("<h4>Key Risk Indicators</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h5>Content Indicators</h5>", unsafe_allow_html=True)
                    
                    # Combine indicators from text and image
                    content_indicators = []
                    
                    if image_prediction == "Fake":
                        content_indicators.append("• Image appears to be manipulated or AI-generated")
                    
                    if text_prediction == "Fake" and text_explanation['key_indicators']:
                        for indicator in text_explanation['key_indicators'][:2]:  # Top 2
                            content_indicators.append(f"• {indicator}")
                    
                    if not content_indicators:
                        content_indicators.append("• No significant content risk indicators detected")
                    
                    for indicator in content_indicators:
                        st.markdown(indicator)
                
                with col2:
                    st.markdown("<h5>Account Indicators</h5>", unsafe_allow_html=True)
                    
                    # Get account indicators
                    account_indicators = []
                    
                    if account_metrics['suspicious_indicators']:
                        for key, value in list(account_metrics['suspicious_indicators'].items())[:2]:  # Top 2
                            account_indicators.append(f"• {key}: {value}")
                    
                    if not account_indicators:
                        account_indicators.append("• No significant account risk indicators detected")
                    
                    for indicator in account_indicators:
                        st.markdown(indicator)
                
                # Display recommendations
                st.markdown("<h4>Recommendations</h4>", unsafe_allow_html=True)
                
                # Combine all recommendations
                all_recommendations = []
                
                # Add recommendations based on image analysis
                if image_prediction == "Fake":
                    all_recommendations.append("• Verify the authenticity of this image through alternate sources")
                
                # Add recommendations from text analysis
                if text_prediction == "Fake":
                    all_recommendations.append("• Exercise caution with this messaging")
                    all_recommendations.append("• Check for verification from trusted sources")
                
                # Add recommendations from account# Add recommendations from account analysis
                if account_metrics['recommendations']:
                    for rec in account_metrics['recommendations'][:2]:  # Top 2 recommendations
                        all_recommendations.append(f"• {rec}")
                
                # General recommendation for all analyses
                if not all_recommendations:
                    all_recommendations.append("• Content appears authentic, but general digital literacy is still recommended")
                
                for rec in all_recommendations:
                    st.markdown(rec)
            
            with result_tabs[1]:
                # Display image analysis results
                if image_prediction == "Real":
                    st.markdown('<div class="result-real">✅ REAL IMAGE</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">⚠️ DEEPFAKE DETECTED</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="confidence">Confidence: {image_confidence:.2f}%</div>', unsafe_allow_html=True)
                
                # Display probability distribution
                fig, ax = plt.subplots(figsize=(10, 3))
                
                # Create bar chart for probabilities
                classes = ['Fake', 'Real']
                probabilities = [
                    100 - image_confidence if image_prediction == "Real" else image_confidence,
                    image_confidence if image_prediction == "Real" else 100 - image_confidence
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
                ax.set_title('Image Prediction Probabilities', fontsize=14, pad=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
            
            with result_tabs[2]:
                # Display text analysis results
                prediction, confidence = text_prediction, text_confidence
                
                if prediction == "Real":
                    st.markdown('<div class="result-real">✅ AUTHENTIC TEXT CONTENT</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">⚠️ FAKE TEXT CONTENT DETECTED</div>', unsafe_allow_html=True)
                    
                st.markdown(f'<div class="confidence">Confidence: {confidence:.2%} ({text_explanation["confidence_level"]})</div>', unsafe_allow_html=True)
                
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
                    if value <= 33:
                        color = gauge_colors[0]  # Green for low risk
                    elif value <= 66:
                        color = gauge_colors[1]  # Yellow for medium risk
                    else:
                        color = gauge_colors[2]  # Red for high risk
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
                ax.text(20, -0.8, "Authentic", fontsize=10)
                ax.text(80, -0.8, "Fake", fontsize=10)
                
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
                
                # Display key indicators
                st.markdown("<h4>Key Indicators</h4>", unsafe_allow_html=True)
                if text_explanation['key_indicators']:
                    for indicator in text_explanation['key_indicators']:
                        st.markdown(f"• {indicator}")
                else:
                    st.markdown("No strong indicators detected.")
                
                # Display preprocessed text
                with st.expander("View Preprocessed Text"):
                    st.code(text_explanation["preprocessed_text"])
            
            with result_tabs[3]:
                # Display account analysis results
                if account_prediction == "Authentic":
                    st.markdown('<div class="result-real">✅ LIKELY AUTHENTIC ACCOUNT</div>', unsafe_allow_html=True)
                elif account_prediction == "Attention":
                    st.markdown('<div class="result-fake" style="background-color: #FEF3C7; color: #D97706;">⚠️ REQUIRES ATTENTION</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-fake">⚠️ SUSPICIOUS ACCOUNT DETECTED</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="confidence">Confidence: {account_metrics["confidence"]:.2f}%</div>', unsafe_allow_html=True)
                
                # Display risk score
                fig, ax = plt.subplots(figsize=(10, 2))
                
                # Create risk gauge
                risk_score = account_metrics['risk_score']
                gauge_colors = ['#10B981', '#FBBF24', '#EF4444']
                bar_height = 0.6
                
                # Draw the gauge bar background
                ax.barh(0, 100, height=bar_height, color='#e5e7eb')
                
                # Draw the gauge value with appropriate color
                if risk_score <= 33:
                    color = gauge_colors[0]  # Green for low risk
                elif risk_score <= 66:
                    color = gauge_colors[1]  # Yellow for medium risk
                else:
                    color = gauge_colors[2]  # Red for high risk
                
                ax.barh(0, risk_score, height=bar_height, color=color)
                
                # Add marker for threshold
                ax.axvline(x=33, color='black', linestyle='--', alpha=0.7)
                ax.axvline(x=66, color='black', linestyle='--', alpha=0.7)
                
                # Add labels
                ax.text(15, -0.8, "Low Risk", fontsize=10)
                ax.text(45, -0.8, "Medium Risk", fontsize=10)
                ax.text(80, -0.8, "High Risk", fontsize=10)
                
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
                
                # Display metrics breakdown
                st.markdown("<h4>Metrics Analysis</h4>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create bar chart for good indicators
                    st.markdown("<h5>Positive Indicators</h5>", unsafe_allow_html=True)
                    for indicator, value in account_metrics['positive_indicators'].items():
                        st.markdown(f"• {indicator}: {value}")
                
                with col2:
                    # Create bar chart for suspicious indicators
                    st.markdown("<h5>Suspicious Indicators</h5>", unsafe_allow_html=True)
                    for indicator, value in account_metrics['suspicious_indicators'].items():
                        st.markdown(f"• {indicator}: {value}")
                
                # Display recommendations
                st.markdown("<h4>Recommendations</h4>", unsafe_allow_html=True)
                for rec in account_metrics['recommendations']:
                    st.markdown(f"• {rec}")

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
        <h4>👤 Account Analysis</h4>
        <p>Evaluate social media account metrics for suspicious patterns.</p>
    </div>
    
    <div class="info-box">
        <h4>🔄 Combined Analysis</h4>
        <p>Analyze multiple elements simultaneously for a comprehensive assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button
    if st.sidebar.button("Reset All Analyses"):
        # Reset session state
        st.session_state.image_analyzed = False
        st.session_state.text_analyzed = False
        st.session_state.combined_analyzed = False
        st.session_state.account_analyzed = False
        st.session_state.image_result = None
        st.session_state.text_result = None
        st.session_state.account_result = None
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
    if st.session_state.image_analyzed or st.session_state.text_analyzed or st.session_state.account_analyzed or st.session_state.combined_analyzed:
        # If analyses have been performed, show the dashboard first
        tabs = st.tabs(["Dashboard", "Image Analysis", "Text Analysis", "Account Analysis", "Combined Analysis"])
        
        with tabs[0]:
            display_dashboard()
            
        with tabs[1]:
            display_image_analysis()
            
        with tabs[2]:
            display_text_analysis()
            
        with tabs[3]:
            display_account_analysis()
            
        with tabs[4]:
            display_combined_analysis()
    else:
        # If no analyses yet, don't show dashboard tab
        tabs = st.tabs(["Image Analysis", "Text Analysis", "Account Analysis", "Combined Analysis"])
        
        with tabs[0]:
            display_image_analysis()
            
        with tabs[1]:
            display_text_analysis()
            
        with tabs[2]:
            display_account_analysis()
            
        with tabs[3]:
            display_combined_analysis()

# Run the application
if __name__ == "__main__":
    main()