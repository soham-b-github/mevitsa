import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import io
import json
import altair as alt
from model import MeViTSA # type: ignore


# ==========================================
# SETUP CLOUD ENVIRONMENT METHOD
# ==========================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress most logs
from huggingface_hub import hf_hub_download
from kaggle.api.kaggle_api_extended import KaggleApi
import shutil

@st.cache_resource
def setup_cloud_environment():
    # -- HuggingFace Models --
    # DOWNLOAD MODELS FROM HUGGINGFACE
    hf_token = st.secrets["HF_TOKEN"]
    hf_repo_id = "soham-b/mevitsa"
    hf_folder_name = "trained-models"
    
    models_to_download = ["T5__best_model.pt", "CLIP__best_model.pt"]
    os.makedirs("./trained", exist_ok=True)
    
    for filename in models_to_download:
        expected_path = os.path.join("./trained", filename)
        
        if not os.path.exists(expected_path):
            print(f"Downloading {filename} from HuggingFace...")
            
            hf_file_path = f"{hf_folder_name}/{filename}"
            
            cached_path = hf_hub_download(
                repo_id=hf_repo_id, 
                filename=hf_file_path,
                token=hf_token
            )
            
            # Move it to the exact location your config-models.json expects
            shutil.copy(cached_path, expected_path)

    # -- Kaggle Dataset --
    # DOWNLOAD DATASET FROM KAGGLE
    os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
    os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]
    
    kaggle_dataset_id = "sohambhattacharyaa/musait" # <-- UPDATE THIS
    csv_target_path = "./dataset/docimsentv1.csv" 
    download_dir = os.path.dirname(csv_target_path)
    
    os.makedirs(download_dir, exist_ok=True)
    
    if not os.path.exists(csv_target_path):
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(kaggle_dataset_id, path=download_dir, unzip=True)
        
    return True

setup_cloud_environment()


# --- Logo ---
logo_path = "./../assets/logo/logo.png"
logo_image = Image.open(logo_path) if os.path.exists(logo_path) else "M"
dark_logo_image = Image.open("./../assets/logo/logo-darkbg.png") if os.path.exists("./../assets/logo/logo-darkbg.png") else None
st.set_page_config(page_title="MeViTSA Method", page_icon=logo_image, layout="wide")

bg_color = "#121212"
text_color = "#FFFFFF"
panel_bg = "#1E1E1E"
panel_text = "#FFFFFF"
uploader_bg = "#262730"
glass_bg = "rgba(255, 255, 255, 0.05)"

# --- Theme colours ---
GOOGLE_BLUE, GOOGLE_RED, GOOGLE_YELLOW, GOOGLE_GREEN = "#4285F4", "#EA4335", "#FBBC05", "#34A853"
text_color, glass_bg, bg_color = "#E0E0E0", "rgba(255, 255, 255, 0.05)", "#121212"

st.markdown(f"""
<style>
    /* 1. Shared Animation for all moving gradients */
    @keyframes move-gradient {{
        0% {{ background-position: 0% 50%; }}
        100% {{ background-position: 200% 50%; }}
    }}

    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    h1, h2, h3 {{ 
        font-family: 'Product Sans', sans-serif; 
        color: {GOOGLE_BLUE}; 
    }}

    /* 2. THE TOP DECORATION BAR (Restored & Fixed) */
    .decoration-bar {{
        height: 6px; 
        width: 100%;
        background: linear-gradient(
            to right, 
            {GOOGLE_BLUE}, {GOOGLE_RED}, {GOOGLE_YELLOW}, {GOOGLE_GREEN}, {GOOGLE_BLUE}
        );
        background-size: 200% 100%;
        animation: move-gradient 3s linear infinite;
        margin-bottom: 20px; 
        border-radius: 3px;
    }}

    /* 3. THE ANIMATED GLASS BUTTON */
    div.stButton > button:first-child {{
        position: relative;
        background: {glass_bg}; 
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        color: {text_color};
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        overflow: hidden;
        z-index: 1;
        font-weight: 600;
        transition: transform 0.2s ease;
    }}

    /* Border logic for the button */
    div.stButton > button:first-child::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 8px; 
        padding: 2px; /* Border thickness */
        background: linear-gradient(
            to right, 
            {GOOGLE_BLUE}, {GOOGLE_RED}, {GOOGLE_YELLOW}, {GOOGLE_GREEN}, {GOOGLE_BLUE}
        );
        background-size: 200% 100%;
        -webkit-mask: 
            linear-gradient(#fff 0 0) content-box, 
            linear-gradient(#fff 0 0);
        -webkit-mask-composite: destination-out;
        mask-composite: exclude;
        animation: move-gradient 3s linear infinite;
        z-index: -1;
    }}

    div.stButton > button:first-child:hover {{
        transform: scale(1.02);
        background: {glass_bg};
        color: {text_color};
    }}
</style>
""", unsafe_allow_html=True)

# --- ADVANCED CSS INJECTION ---
st.markdown(f"""
<style>
    /* Global Styles */
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}

    /* Sidebar / Panel Styles */
    [data-testid="stSidebar"] {{
        background-color: {panel_bg};
        color: {panel_text};
    }}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{
        color: {panel_text} !important;
    }}

    /* Drag and Drop Block (File Uploader) */
    [data-testid="stFileUploader"] {{
        background-color: {uploader_bg};
        border-radius: 10px;
        padding: 10px;
    }}
    [data-testid="stFileUploader"] section {{
        background-color: {uploader_bg} !important;
        color: {text_color} !important;
    }}

    /* Top Decoration Bar */
    @keyframes move-gradient {{
        0% {{ background-position: 0% 50%; }}
        100% {{ background-position: 200% 50%; }}
    }}
    .decoration-bar {{
        height: 6px; width: 100%;
        background: linear-gradient(to right, {GOOGLE_BLUE}, {GOOGLE_RED}, {GOOGLE_YELLOW}, {GOOGLE_GREEN}, {GOOGLE_BLUE});
        background-size: 200% 100%;
        animation: move-gradient 3s linear infinite;
        margin-bottom: 20px; border-radius: 3px;
    }}

    /* Animated Glass Button */
    div.stButton > button:first-child {{
        position: relative;
        background: {glass_bg}; 
        backdrop-filter: blur(10px);
        color: {text_color};
        border: none; padding: 10px 24px; border-radius: 8px;
        overflow: hidden; z-index: 1; font-weight: 600;
    }}
    div.stButton > button:first-child::before {{
        content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        border-radius: 8px; padding: 2px;
        background: linear-gradient(to right, {GOOGLE_BLUE}, {GOOGLE_RED}, {GOOGLE_YELLOW}, {GOOGLE_GREEN}, {GOOGLE_BLUE});
        background-size: 200% 100%;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: destination-out; mask-composite: exclude;
        animation: move-gradient 3s linear infinite; z-index: -1;
    }}
</style>
""", unsafe_allow_html=True)

# --- ENGINE INITIALIZATION ---
@st.cache_resource(show_spinner=False)
def get_mevitsa_engine():
    with open("./../configs/config-models.json") as f:
        config = json.load(f)
    return MeViTSA(config)


if 'engine_loaded' not in st.session_state:
    with st.status("Initializing MeViTSA engine...", expanded=True) as status:
        st.write("Connecting to neural compute units...")
        # Simulate a small delay or just run the heavy task
        try:
            engine = get_mevitsa_engine()
            st.write("Models loaded successfully !!")
            # st.write("Configuring UI themes...!")
            status.update(label="System Ready!", state="complete", expanded=False)
            st.session_state.engine_loaded = True
        except Exception as e:
            status.update(label="Initialization Failed", state="error")
            st.error(f"Error loading models: {e}")
            st.stop()
else:
    # If already cached, just grab the engine silently
    engine = get_mevitsa_engine()

# --- HEADER ---
st.markdown('<div class="decoration-bar"></div>', unsafe_allow_html=True)
st.title("MeViTSA: Multimodal Sentiment Analysis framework")
st.markdown("**Method: MeViTSA (Multimodal ensemble approach for Visuals Integrated Text Data for Sentiment Analysis)**")

# --- SIDEBAR ---
with st.sidebar:
    st.image(dark_logo_image, width=225)
    st.sidebar.header("Configuration")
    use_auto_alpha = st.sidebar.checkbox("Auto-calculate alpha (text area)", value=True)
    alpha_input = None if use_auto_alpha else st.sidebar.slider("Manual fusion weight (alpha)", 0.0, 1.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Advanced framework settings")
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
        st.image(image_pil, caption='Input image', use_container_width=True)

    with col2:
        st.subheader("Analysis results")
        if st.button("Run MeViTSA analysis", type="primary"):
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
                    st.success(f"Text source: {res['source']}")
                    st.info(f"**Text content:** \"{res['text']}\"")
                    
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
                    # st.bar_chart(chart_df.set_index("Sentiment"))
                    # Define the soft color palette
                    # Using Hex codes for Soft Red, Soft Yellow, and Soft Green
                    color_scale = alt.Chart(chart_df).mark_bar().encode(
                        x=alt.X('Sentiment:N', sort=None),
                        y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('Sentiment:N', scale=alt.Scale(
                            domain=['Negative', 'Neutral', 'Positive'],
                            range=["#ff5050", '#fff9ae', '#90ee90']  # soft red, soft yellow, soft green
                        ), legend=None)
                    ).properties(
                        height=300
                    ).configure_axis(
                        labelColor=text_color,
                        titleColor=text_color,
                        gridColor=f"{text_color}33" # Semi-transparent grid lines
                    ).configure_view(
                        strokeOpacity=0 # Remove chart border
                    )

                    st.altair_chart(color_scale, use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
