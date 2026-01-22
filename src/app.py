import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import io
import json
from model import MeVITSA # type: ignore

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MeViTSA Method", page_icon="🧠", layout="wide")

# --- STYLING (Preserved) ---
GOOGLE_BLUE, GOOGLE_RED, GOOGLE_YELLOW, GOOGLE_GREEN = "#4285F4", "#EA4335", "#FBBC05", "#34A853"

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

# --- ENGINE INITIALIZATION ---
@st.cache_resource(show_spinner=False)
def get_mevitsa_engine():
    with open("./../configs/config-models.json") as f:
        config = json.load(f)
    return MeVITSA(config)

try:
    with st.spinner("Loading models for MeVITSA framework..."):
        engine = get_mevitsa_engine()
    st.sidebar.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- HEADER ---
st.markdown('<div class="decoration-bar"></div>', unsafe_allow_html=True)
st.title("MeViTSA: Multimodal Sentiment Analysis")
st.markdown("**Method: MeVITSA (Multimodal ensemble approach for Visuals Integrated Text Data for Sentiment Analysis)**")

# --- SIDEBAR (Preserved + New Settings) ---
st.sidebar.header("Configuration")
use_auto_alpha = st.sidebar.checkbox("Auto-calculate alpha (text area)", value=True)
alpha_input = None if use_auto_alpha else st.sidebar.slider("Manual fusion weight (alpha)", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Engine Settings")
gcv_on = st.sidebar.checkbox("Use GCV API", value=False, help="Enable live Google Cloud Vision OCR")
ft_on = st.sidebar.checkbox("Fault Tolerance (BLIP)", value=True, help="Use BLIP if OCR fails")

# --- MAIN UI ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "JPG"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    img_byte_arr = io.BytesIO()
    image_pil.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()

    with col1:
        st.image(image_pil, caption='Input Image', use_container_width=True)

    with col2:
        st.subheader("Analysis Results")
        if st.button("Run MeVITSA Analysis", type="primary"):
            with st.spinner('Running inference...'):
                try:
                    # UPDATED CALL: Pass UI flags and filename
                    res = engine.analyze(
                        image_pil, 
                        image_bytes, 
                        image_filename=uploaded_file.name,
                        manual_alpha=alpha_input,
                        gcv_api=gcv_on,
                        ft=ft_on
                    )
                    
                    # Dynamic Status display based on res["source"]
                    st.success(f"✅ Method: {res['source']}")
                    st.info(f"**Text Content:** \"{res['text']}\"")
                    
                    # Sentiment Display (Preserved Design)
                    color_map = {"Negative": GOOGLE_RED, "Neutral": GOOGLE_YELLOW, "Positive": GOOGLE_GREEN}
                    res_color = color_map.get(res["label"], GOOGLE_BLUE)
                    
                    st.markdown(f"""
                    <div style="border-left: 5px solid {res_color}; padding-left: 15px; margin-top: 20px;">
                        <h2 style="color: {res_color}; margin:0;">{res["label"]}</h2>
                        <p style="font-size: 18px; color: #5f6368;">Confidence: <b>{np.max(res['probs'])*100:.2f}%</b></p>
                        <p style="font-size: 14px;">Alpha Used: <b>{res['alpha']:.2f}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Charting (Preserved Design)
                    chart_df = pd.DataFrame({
                        "Sentiment": ["Negative", "Neutral", "Positive"],
                        "Probability": res["probs"]
                    })
                    st.bar_chart(chart_df.set_index("Sentiment"))
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")