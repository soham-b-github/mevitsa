import streamlit as st
import torch
import numpy as np
from PIL import Image
from google.cloud import vision
import os
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MeViTSA",
    page_icon="🧠",
    layout="wide"
)

# --- GOOGLE THEME COLORS & CSS ---
GOOGLE_BLUE = "#4285F4"
GOOGLE_RED = "#EA4335"
GOOGLE_YELLOW = "#FBBC05"
GOOGLE_GREEN = "#34A853"

st.markdown(f"""
<style>
    /* Main Headers */
    h1, h2, h3 {{
        font-family: 'Product Sans', 'Roboto', sans-serif;
        color: {GOOGLE_BLUE};
    }}
    
    /* Top Decoration Bar */
    .decoration-bar {{
        height: 6px;
        width: 100%;
        background: linear-gradient(to right, 
            {GOOGLE_BLUE} 25%, 
            {GOOGLE_RED} 25% 50%, 
            {GOOGLE_YELLOW} 50% 75%, 
            {GOOGLE_GREEN} 75%);
        margin-bottom: 20px;
        border-radius: 3px;
    }}
    
    /* Customizing the Sidebar Slider */
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {{
        background-color: {GOOGLE_BLUE};
        box-shadow: rgb(14 123 255 / 20%) 0px 0px 0px 0.2rem;
    }}
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] {{
        color: {GOOGLE_BLUE};
    }}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="decoration-bar"></div>', unsafe_allow_html=True)
st.markdown(f"""
<h1>
    <span style='color:{GOOGLE_BLUE}'>M</span>
    <span style='color:{GOOGLE_RED}'>e</span>
    <span style='color:{GOOGLE_YELLOW}'>V</span>
    <span style='color:{GOOGLE_BLUE}'>i</span>
    <span style='color:{GOOGLE_GREEN}'>T</span>
    <span style='color:{GOOGLE_RED}'>S</span>
    <span style='color:{GOOGLE_BLUE}'>A</span>
</h1>
""", unsafe_allow_html=True)
st.markdown("**Multimodal Sentiment Analysis:** Robust architecture for images with embedded text.")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
alpha = st.sidebar.slider(
    "Fusion Weight (Alpha)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.5, 
    help="1.0 = Text only, 0.0 = Visual only."
)

# --- 1. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_models():
    """
    Load your pre-trained models here.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Placeholder loading logic
    return {"device": device}

models = load_models()

# --- 2. HELPER FUNCTIONS ---

def google_ocr_process(image_content):
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        return None
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    if texts:
        return texts[0].description.replace('\n', ' ')
    return None

def generate_caption(image, models):
    # Placeholder for BLIP logic
    return "Generated caption: A screenshot of a product."

def get_visual_sentiment(image, models):
    # Placeholder: [Negative, Neutral, Positive]
    return np.array([0.1, 0.2, 0.7]) 

def get_textual_sentiment(text, models):
    # Placeholder: [Negative, Neutral, Positive]
    return np.array([0.05, 0.15, 0.8])

# --- 3. MAIN LOGIC ---

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption='Input Image', use_container_width=True)
        
        # Convert for OCR
        import io
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format)
        image_bytes = img_byte_arr.getvalue()

    with col2:
        st.subheader("Processing pipeline")
        
        with st.spinner('Running MeViTSA Pipeline...'):
            
            # Step 1: Textual Channel
            extracted_text = google_ocr_process(image_bytes)
            source = ""
            
            if extracted_text and len(extracted_text.strip()) > 2:
                st.success("✅ OCR Successful")
                final_text_input = extracted_text
                source = "OCR"
            else:
                # Using Yellow for warning/fault tolerance
                st.markdown(f"<div style='color:{GOOGLE_YELLOW}; font-weight:bold;'>⚠️ OCR low/empty. Activating BLIP Fault Tolerance.</div>", unsafe_allow_html=True)
                final_text_input = generate_caption(image, models)
                source = "BLIP Caption"
            
            st.info(f"**Text Input ({source}):** {final_text_input}")
            
            # Step 2: Inference
            p_visual = get_visual_sentiment(image, models)
            p_text = get_textual_sentiment(final_text_input, models)
            
            # Step 3: Late Fusion
            p_final = (alpha * p_text) + ((1 - alpha) * p_visual)
            
            # Determine Class
            classes = ["Negative", "Neutral", "Positive"]
            class_idx = np.argmax(p_final)
            predicted_class = classes[class_idx]
            confidence = np.max(p_final)

        # --- Step 4: Display Results (Google Colors) ---
        st.divider()
        st.subheader("Classification Result")
        
        # Determine color based on sentiment
        if predicted_class == "Negative":
            res_color = GOOGLE_RED
        elif predicted_class == "Neutral":
            res_color = GOOGLE_YELLOW
        else:
            res_color = GOOGLE_GREEN

        # Display Big Result using HTML for specific color control
        st.markdown(f"""
        <div style="border-left: 5px solid {res_color}; padding-left: 15px;">
            <h2 style="color: {res_color}; margin:0;">{predicted_class}</h2>
            <p style="font-size: 18px; color: #5f6368;">Confidence: <b>{confidence*100:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Breakdown Chart
        st.write("### Probability Distribution")
        
        # Create a DataFrame for the chart to allow color mapping
        chart_data = pd.DataFrame({
            "Sentiment": classes,
            "Probability": p_final,
        })
        
        # Streamlit's basic bar chart uses one color. We set it to Google Blue.
        st.bar_chart(chart_data.set_index("Sentiment"), color=GOOGLE_BLUE)
		
        with st.expander("See Channel Contributions"):
            st.write(f"**Visual Channel:** {p_visual}")
            st.write(f"**Textual Channel:** {p_text}")

else:
    st.info("Please upload an image to begin analysis.")
