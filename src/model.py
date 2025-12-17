import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from PIL import Image
from google.cloud import vision

# 1. Define a local folder for models
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# 2. Tell Hugging Face to use this folder
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip

# --- CONFIGURATION ---
# UPDATE THESE PATHS TO YOUR LOCAL PATHS
TEXT_MODEL_NAME = "t5-base"  # Or your specific fine-tuned path
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_WEIGHTS_PATH = "./../../best_models_experimentation__DO_NOT_DELETE/VSA/CLIP__best_model.pt" # Update this
TEXT_WEIGHTS_PATH = "./../../best_models_experimentation__DO_NOT_DELETE/TSA/T5__best_model.pt" # Update this

MAX_LENGTH = 128
SENTIMENTS_TO_LABELS = {'negative':0,'neutral':1,'positive':2}
LABELS_TO_SENTIMENTS = {0: 'Negative', 1: 'Neutral', 2: "Positive"}

csv_filepath = "./../file/dataset_docimsentv1-info/docimsentv1.csv"
df = pd.read_csv(csv_filepath)
all_filenames = df["filename"]
all_ocr = df["OCR"]

# --- CLIP CLASSIFIER WRAPPER ---
class CLIPClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        image = image.to(dtype=self.encoder.conv1.weight.dtype)
        with torch.no_grad():
            x = self.encoder(image)
        x = x.float()
        return self.classifier(x)

# --- MODEL LOADER ---
def load_core_models():
    """
    Loads all models onto the device once to be cached by Streamlit.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models on {device}...")

    # 1. Load BLIP (Captioning)
    blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME, cache_dir=cache_dir)
    blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME, cache_dir=cache_dir).to(device)

    # 2. Load Text Model (DistilRoBERTa/T5)
    # Note: Ensure num_labels matches your training (e.g., 3 for Neg/Neu/Pos)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir=cache_dir)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, num_labels=3, cache_dir=cache_dir)
    
    # Load weights if they exist, otherwise use base (warning: base model won't predict sentiment correctly without fine-tuning)
    if os.path.exists(TEXT_WEIGHTS_PATH):
        text_model.load_state_dict(torch.load(TEXT_WEIGHTS_PATH, map_location=device))
    text_model.to(device)
    text_model.eval()

    # 3. Load CLIP (Visual)
    clip_path = os.path.join(cache_dir, "ViT-B-32.pt")
    clip_base, image_preprocess = clip.load(CLIP_MODEL_NAME, device=device, download_root=cache_dir)
    # We need the embed_dim from the visual part of CLIP (usually 512 for ViT-B/32)
    embed_dim = clip_base.visual.output_dim 
    
    image_model = CLIPClassifier(clip_base.visual, embed_dim=embed_dim, num_classes=3).to(device)
    if os.path.exists(CLIP_WEIGHTS_PATH):
        image_model.load_state_dict(torch.load(CLIP_WEIGHTS_PATH, map_location=device))
    image_model.eval()

    return {
        "device": device,
        "blip_processor": blip_processor,
        "blip_model": blip_model,
        "text_tokenizer": text_tokenizer,
        "text_model": text_model,
        "image_preprocess": image_preprocess,
        "image_model": image_model
    }

# --- HELPER: GOOGLE VISION OCR ---
def get_ocr_text(image_content):
    """
    Performs OCR using Google Vision API.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    if texts:
        # texts[0] is the full text, subsequent indices are words
        return texts[0].description, texts[0] 
    return None, None
 
 
def check_text_detection_v1(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    client = vision.ImageAnnotatorClient()
    
    image = vision.Image(content=content)
    text = None
    
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0] if texts else False


def check_text_detection_v2(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    client = vision.ImageAnnotatorClient()
    
    image = vision.Image(content=content)
    text = None
    
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts if texts else False


# --- HELPER: ALPHA CALCULATION (Simplified v1) ---
def calculate_alpha(ocr_annotation, img_width, img_height):
    """
    Calculates the area of text relative to image area.
    """
    if not ocr_annotation:
        return 0.5 # Default if no text found via OCR

    img_area = img_width * img_height
    
    # Calculate bounding box of the full text
    vertices = ocr_annotation.bounding_poly.vertices
    x_values = [v.x for v in vertices]
    y_values = [v.y for v in vertices]
    
    txt_width = max(x_values) - min(x_values)
    txt_height = max(y_values) - min(y_values)
    
    txt_area = txt_width * txt_height
    alpha = txt_area / img_area
    
    # Clip alpha to reasonable bounds (e.g., don't let it be 0 or 1 purely based on size)
    return min(max(alpha, 0.2), 0.8)
    

def get_alpha_v1(texts, img_area):
    text = texts.description
    bounding_poly = texts.bounding_poly
    coordinates = []
    for vertex in bounding_poly.vertices:
        coordinates.append({"x": vertex.x, "y": vertex.y})
    
    x_values = [point['x'] for point in coordinates]
    y_values = [point['y'] for point in coordinates]
    txt_width = max(x_values) - min(x_values)
    txt_height = max(y_values) - min(y_values)

    txt_area = txt_width*txt_height
    txt_weightage = txt_area/img_area
    return txt_weightage
    

def get_alpha_v2(texts, img_area):
   total_text_area = 0
   for t in texts[1:]:
       vertices = [(v.x, v.y) for v in t.bounding_poly.vertices]
       text_area = polygon_area(vertices)
       total_text_area += text_area
       print(f"Word: {t.description}, Area: {text_area}")
   
   print("Total Area:", total_text_area)
   alpha = total_text_area/img_area
   return alpha

# --- MAIN INFERENCE PIPELINE (MELT) ---
# --- MAIN INFERENCE PIPELINE (MELT) ---
def melt_analysis(image_filename, image_pil, image_bytes, models_dict, manual_alpha=None):
    """
    The MELT Method:
    1. Extract Text (OCR from CSV) -> if failed, Generate Caption (BLIP).
    2. Get Text Sentiment (RoBERTa/T5).
    3. Get Image Sentiment (CLIP).
    4. Fuse using Alpha (Area-based or Manual).
    """
    device = models_dict["device"]
    
    # 1. VISUAL PIPELINE
    # ------------------
    image_tensor = models_dict["image_preprocess"](image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        visual_logits = models_dict["image_model"](image_tensor)
        visual_probs = F.softmax(visual_logits, dim=1).cpu().numpy().flatten()
    
    # 2. TEXT PIPELINE
    # ----------------
    ocr_text = ""
    final_text = ""
    source = "BLIP" # Default source
    
    # Check if image exists in the CSV data
    # Note: using .values ensures we check the content, not the index
    if image_filename in all_filenames.values:
        matched_row = df.loc[df["filename"] == image_filename].iloc[0]
        ocr_text = matched_row["OCR"]
    
    # Logic: If OCR exists and is not empty/NaN, use it. Otherwise, use BLIP.
    if pd.notna(ocr_text) and str(ocr_text).strip() != "":
        final_text = str(ocr_text)
        source = "OCR"
    else:
        # Fault Tolerance: Generate Caption via BLIP
        source = "BLIP"
        inputs = models_dict["blip_processor"](images=image_pil, return_tensors="pt").to(device)
        out = models_dict["blip_model"].generate(**inputs)
        final_text = models_dict["blip_processor"].decode(out[0], skip_special_tokens=True)
    
    # Tokenize and Predict
    inputs = models_dict["text_tokenizer"](
        final_text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=MAX_LENGTH
    ).to(device)
    
    with torch.no_grad():
        text_outputs = models_dict["text_model"](**inputs)
        text_probs = F.softmax(text_outputs.logits, dim=1).cpu().numpy().flatten()

    # 3. FUSION (MELT LOGIC)
    # ----------------------
    if manual_alpha is not None:
        alpha = manual_alpha
    else:
        # Note: Since we are reading from CSV, we might not have 'ocr_annotation' 
        # for area calculation unless it's also saved in the CSV. 
        # For now, we default to 0.5 if using BLIP or if area data is missing.
        alpha = 0.5 

    # Weighted Average
    final_probs = (alpha * text_probs) + ((1 - alpha) * visual_probs)
    pred_idx = np.argmax(final_probs)
    
    return {
        "final_class": LABELS_TO_SENTIMENTS[pred_idx],
        "final_probs": final_probs,
        "text_probs": text_probs,
        "visual_probs": visual_probs,
        "text_content": final_text,
        "text_source": source,
        "alpha_used": alpha
    }
