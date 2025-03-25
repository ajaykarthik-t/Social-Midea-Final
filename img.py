import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Deep Fake Detection",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .result-real {
        font-size: 2rem;
        color: #10B981;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1FAE5;
    }
    .result-fake {
        font-size: 2rem;
        color: #EF4444;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FEE2E2;
    }
    .confidence {
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Deep Fake Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses a deep learning model to detect whether an uploaded image is real or a deepfake.
Upload your image to get started.
""")

# Create a function to load the model
@st.cache_resource
def load_detection_model():
    try:
        return load_model('my_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Create a function to make predictions
def predict_image(img_array, model):
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

# Main application logic
def main():
    # Load the model
    model = load_detection_model()
    
    if model is None:
        st.warning("Please make sure the model file 'my_model.keras' is in the same directory as this script.")
        return
    
    # File uploader
    st.markdown('<h2 class="sub-header">Upload an Image</h2>', unsafe_allow_html=True)
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
            if st.button("Detect Deepfake"):
                with st.spinner("Processing image..."):
                    # Resize image to match model's expected input
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized)
                    
                    # If image has 4 channels (RGBA), convert to 3 channels (RGB)
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                    
                    # Make prediction
                    predicted_class, confidence = predict_image(img_array, model)
                    
                    # Display the result with appropriate styling
                    st.markdown("<h2 class='sub-header'>Detection Result:</h2>", unsafe_allow_html=True)
                    
                    if predicted_class == "Real":
                        st.markdown(f"<div class='result-real'>✅ REAL IMAGE</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-fake'>⚠️ DEEPFAKE DETECTED</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"<div class='confidence'>Confidence: {confidence:.2f}%</div>", unsafe_allow_html=True)
                    
                    # Display probability distribution
                    st.markdown("<h2 class='sub-header'>Probability Distribution:</h2>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(8, 2))
                    
                    # Create bar chart for probabilities
                    classes = ['Fake', 'Real']
                    probabilities = [
                        100 - confidence if predicted_class == "Real" else confidence,
                        confidence if predicted_class == "Real" else 100 - confidence
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
                    
        except Exception as e:
            st.error(f"Error processing the image: {e}")
    
    # Information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">About Deep Fake Detection</h2>', unsafe_allow_html=True)
    st.markdown("""
    This application uses a deep learning model trained to distinguish between real images and deepfakes.
    
    A deepfake is a type of synthetic media where a person's likeness is replaced with someone else's using artificial intelligence.
    This technology can be misused to create misleading content, which is why detection tools are important.
    
    Our model analyzes various aspects of the image to identify signs of manipulation that are characteristic of deepfakes.
    """)

if __name__ == "__main__":
    main()