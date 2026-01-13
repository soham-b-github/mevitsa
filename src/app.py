import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import io

# Import the backend logic
import model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MeViTSA Method",
    page_icon="🧠",
    layout="wide"
)

# --- STYLING (Same as before) ---
GOOGLE_BLUE = "#4285F4"
GOOGLE_RED = "#EA4335"
GOOGLE_YELLOW = "#FBBC05"
GOOGLE_GREEN = "#34A853"

st.markdown(f"""
<style>
    h1, h2, h3 {{ font-family: 'Product Sans', sans-serif; color: {GOOGLE_BLUE}; }}
    .decoration-bar {{
        height: 6px; width: 100%;
        background: linear-gradient(to right, {GOOGLE_BLUE}, {GOOGLE_RED}, {GOOGLE_YELLOW}, {GOOGLE_GREEN});
        margin-bottom: 20px; border-radius: 3px;
    }}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<div class="decoration-bar"></div>', unsafe_allow_html=True)
st.title("MeViTSA: Multimodal Sentiment Analysis")
st.markdown("**Method: MeVITSA (Multimodal ensemble approach for Visuals Integrated Text Data for Sentiment Analysis)**")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
use_auto_alpha = st.sidebar.checkbox("Auto-calculate alpha (text area)", value=True)

if not use_auto_alpha:
    alpha_input = st.sidebar.slider("Manual Fusion Weight (Alpha)", 0.0, 1.0, 0.5)
else:
    alpha_input = None  # Signal to backend to calculate it

# --- LOAD MODELS (Cached) ---
# ~ @st.cache_resource
@st.cache_resource(show_spinner=False)
def get_pipeline():
	# This print statement should only appear in your terminal ONCE
    print("DEBUG: Loading models from disk...")
    return model.load_core_models()

try:
    with st.spinner("Loading MeVITSA Models... (this may take a minute)"):
        models = get_pipeline()
    st.sidebar.success("Models Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "JPG"])
image_filename = ""

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image_filename = uploaded_file.name
    # Prepare Image
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    # Prepare Bytes for OCR
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='JPEG') # Convert to JPEG bytes
    image_bytes = img_byte_arr.getvalue()

    with col1:
        st.image(image_pil, caption='Input Image', use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        
        if st.button("Run MeVITSA Analysis", type="primary"):
            with st.spinner('Running inference...'):
                try:
                    # CALL THE BACKEND
                    results = model.mevitsa_analysis(
						image_filename,
                        image_pil, 
                        image_bytes, 
                        models, 
                        manual_alpha=alpha_input,
                        gcv_api=0
                    )
                    
                    # EXTRACT RESULTS
                    pred_class = results["final_class"]
                    conf = np.max(results["final_probs"])
                    text_source = results["text_source"]
                    extracted_text = results["text_content"]
                    
                    # DISPLAY
                    if text_source == "OCR":
                        st.success("✅ Text Detected via OCR")
                    else:
                        st.warning("⚠️ No Text Detected. Generated Caption via BLIP.")
                        
                    st.info(f"**Text Content:** \"{extracted_text}\"")
                    
                    # Sentiment Display
                    color_map = {"Negative": GOOGLE_RED, "Neutral": GOOGLE_YELLOW, "Positive": GOOGLE_GREEN}
                    res_color = color_map.get(pred_class, GOOGLE_BLUE)
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {res_color}; padding-left: 15px; margin-top: 20px;">
                        <h2 style="color: {res_color}; margin:0;">{pred_class}</h2>
                        <p style="font-size: 18px; color: #5f6368;">Confidence: <b>{conf*100:.2f}%</b></p>
                        <p style="font-size: 14px;">Alpha Used: <b>{results['alpha_used']:.2f}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    chart_df = pd.DataFrame({
                        "Sentiment": ["Negative", "Neutral", "Positive"],
                        "Probability": results["final_probs"]
                    })
                    st.bar_chart(chart_df.set_index("Sentiment"))
                    
                    with st.expander("Detailed Probabilities"):
                        st.write("Visual Channel:", results["visual_probs"])
                        st.write("Textual Channel:", results["text_probs"])
                        
                except Exception as e:
                    st.error(f"An error occurred during inference: {e}")
