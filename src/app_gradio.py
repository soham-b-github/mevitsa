import spaces
import torch
import gradio as gr
import numpy as np
from PIL import Image
import pandas as pd
import io
import json
import altair as alt
import os
import shutil
from huggingface_hub import hf_hub_download
from model import MeViTSA # type: ignore

# ==========================================
# SETUP CLOUD ENVIRONMENT METHOD
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress most logs

def setup_cloud_environment():
    # -- HuggingFace Models --
    # DOWNLOAD MODELS FROM HUGGINGFACE (Using env variable instead of st.secrets)
    hf_token = os.environ.get("HF_TOKEN")
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
            
            shutil.copy(cached_path, expected_path)

    # -- Download CSV Dataset --
    csv_target_path = "./dataset/docimsentv1_features.csv"
    download_dir = os.path.dirname(csv_target_path)
    os.makedirs(download_dir, exist_ok=True)

    if not os.path.exists(csv_target_path):
        print("Downloading docimsentv1.csv from HuggingFace...")
        
        hf_csv_path = "docimsentv1_features.csv"
        
        cached_csv = hf_hub_download(
            repo_id=hf_repo_id, 
            filename=hf_csv_path, 
            token=hf_token
        )
        shutil.copy(cached_csv, csv_target_path)
        
    return True

print("Initializing environment...")
setup_cloud_environment()


# --- ENGINE INITIALIZATION ---
def get_mevitsa_engine():
    with open("configs/config-models.json") as f:
        config = json.load(f)
    return MeViTSA(config)

print("Loading MeViTSA engine...")
engine = get_mevitsa_engine()
print("System Ready!")

# --- ADVANCED CSS INJECTION ---
GOOGLE_BLUE, GOOGLE_RED, GOOGLE_YELLOW, GOOGLE_GREEN = "#4285F4", "#EA4335", "#FBBC05", "#34A853"

custom_css = f"""
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
"""

# Helper function to process the file upload and extract the real filename.
def handle_upload(uploaded_file):
    if uploaded_file is None:
        return None, "No file uploaded"
    original_name = os.path.basename(uploaded_file.name)
    return uploaded_file.name, original_name

# --- INFERENCE FUNCTION ---
@spaces.GPU
def analyze_image(img_path, original_filename, use_auto_alpha, alpha_input, gcv_on, ft_on):
    if img_path is None:
        return "No Image Provided", "", "<div style='color:red;'>Please upload an image first.</div>", None

    try:
        # Load and convert image
        image_pil = Image.open(img_path).convert("RGB")
        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format='JPEG')
        image_bytes = img_byte_arr.getvalue()
        # image_filename = os.path.basename(img_path)
        image_filename = original_filename

        # Resolve alpha
        manual_alpha = None if use_auto_alpha else alpha_input

        # Run inference
        res = engine.analyze(
            image_pil, 
            image_bytes, 
            image_filename=image_filename,
            manual_alpha=manual_alpha,
            gcv_api=gcv_on,
            ft=ft_on
        )

        text_source = f"Text source: {res['source']}"
        text_content = f"Text content: \"{res['text']}\""

        # Sentiment Display
        color_map = {"Negative": GOOGLE_RED, "Neutral": GOOGLE_YELLOW, "Positive": GOOGLE_GREEN}
        res_color = color_map.get(res["label"], GOOGLE_BLUE)
        
        html_out = f"""
        <div style="border-left: 5px solid {res_color}; padding-left: 15px; margin-top: 20px;">
            <h2 style="color: {res_color}; margin:0;">{res["label"]}</h2>
            <p style="font-size: 18px; color: #5f6368;">Confidence: <b>{np.max(res['probs'])*100:.2f}%</b></p>
            <p style="font-size: 14px; color: #5f6368;">Alpha Used: <b>{res['alpha']:.2f}</b></p>
        </div>
        """
        
        # Charting
        chart_df = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Probability": res["probs"]
        })
        
        text_color = "#E0E0E0"
        color_scale = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Sentiment:N', sort=None),
            y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('Sentiment:N', scale=alt.Scale(
                domain=['Negative', 'Neutral', 'Positive'],
                range=["#ff5050", '#fff9ae', '#90ee90']
            ), legend=None)
        ).properties(
            height=300
        ).configure_axis(
            labelColor=text_color,
            titleColor=text_color,
            gridColor=f"{text_color}33"
        ).configure_view(
            strokeOpacity=0
        )
        confidences = {
            "Negative": float(res["probs"][0]),
            "Neutral": float(res["probs"][1]),
            "Positive": float(res["probs"][2])
        }

        # --- Custom HTML Progress Bars ---
        sentiment_data = [
            ("Negative", float(res["probs"][0]), "#EA4335"), # Google Red
            ("Neutral", float(res["probs"][1]), "#FBBC05"),  # Google Yellow
            ("Positive", float(res["probs"][2]), "#34A853")  # Google Green
        ]
        
        bars_html = "<div style='display: flex; flex-direction: column; gap: 12px; margin-top: 15px;'>"
        for label, prob, color in sentiment_data:
            pct = prob * 100
            bars_html += f"""
            <div>
                <div style='display: flex; justify-content: space-between; font-size: 14px; font-weight: 600; margin-bottom: 4px;'>
                    <span style='color: {color};'>{label}</span>
                    <span style='color: #5f6368;'>{pct:.1f}%</span>
                </div>
                <div style='width: 100%; background-color: #e5e7eb; border-radius: 999px; height: 8px;'>
                    <div style='width: {pct}%; background-color: {color}; height: 100%; border-radius: 999px;'></div>
                </div>
            </div>
            """
        bars_html += "</div>"
        
        # return text_source, text_content, html_out, color_scale
        # return text_source, text_content, html_out, confidences
        return text_source, text_content, html_out, bars_html


    except Exception as e:
        return "Error", f"An error occurred: {str(e)}", f"<div style='color:red;'>Error during inference.</div>", None

# --- UI LAYOUT ---
def toggle_alpha_visibility(auto_alpha):
    return gr.update(visible=not auto_alpha)

with gr.Blocks() as app:
    gr.HTML('<div class="decoration-bar"></div>')
    gr.Markdown("# MeViTSA: Multimodal Sentiment Analysis framework")
    gr.Markdown("## Method: MeViTSA (Multimodal Ensemble approach for Visuals Integrated Text Data for Sentiment Analysis)")

    with gr.Row():
        # SIDEBAR
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### Configuration")
            use_auto_alpha = gr.Checkbox(label="Auto-calculate alpha (text area)", value=True)
            alpha_input = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Manual fusion weight (alpha)", visible=False)
            
            # Link checkbox to slider visibility
            use_auto_alpha.change(fn=toggle_alpha_visibility, inputs=use_auto_alpha, outputs=alpha_input)
            
            gr.Markdown("---")
            gr.Markdown("### Advanced framework settings")
            gcv_on = gr.Checkbox(label="Use GCV API", value=False, info="Enable live Google Cloud Vision OCR")
            ft_on = gr.Checkbox(label="Fault Tolerance (BLIP)", value=True, info="Use BLIP if OCR fails")
            
        # MAIN UI
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    image_upload = gr.UploadButton("📁 Upload an Image", file_types=["image"], variant="primary")
                    captured_filename = gr.Textbox(label="Original Filename", interactive=False, visible=True) # Set visible=False later if you want to hide it
                    image_preview = gr.Image(type="filepath", label="Image Preview", interactive=False)
                    
                    # image_input = gr.Image(type="filepath", label="Upload an image")
                    analyze_btn = gr.Button("Run MeViTSA analysis", variant="primary")
                    
                with gr.Column():
                    gr.Markdown("### Analysis results")
                    text_source_out = gr.Textbox(label="Status", interactive=False)
                    text_content_out = gr.Textbox(label="Detected Text", interactive=False)
                    sentiment_out = gr.HTML()
                    # plot_out = gr.Plot(label="Probabilities")
                    # plot_out = gr.Label(label="Sentiment Probabilities")
                    plot_out = gr.HTML()

    image_upload.upload(
        fn=handle_upload, 
        inputs=[image_upload], 
        outputs=[image_preview, captured_filename]
    )

    # Connect button to function
    # analyze_btn.click(
    #     fn=analyze_image,
    #     inputs=[image_input, use_auto_alpha, alpha_input, gcv_on, ft_on],
    #     outputs=[text_source_out, text_content_out, sentiment_out, plot_out]
    # )
    # Connect button to function
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_preview, captured_filename, use_auto_alpha, alpha_input, gcv_on, ft_on],
        outputs=[text_source_out, text_content_out, sentiment_out, plot_out]
    )


if __name__ == "__main__":
    app.launch(css=custom_css, theme=gr.themes.Monochrome())
