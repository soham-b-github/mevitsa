import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import io
from tqdm.notebook import tqdm

from PIL import Image
from google.cloud import vision

# 1. Define a local folder for models
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)

# 2. Tell Hugging Face to use this folder
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# 3. It is recommended to make the following changes using terminal instead of code
# The reason behind this is, if the json key is used here in the code, 
# it is not safe, as the code might be shared between individual.
# export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keys/service-account-key.json"

# Otherwise, use the following the inside this code:
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/path/to/your/keys/service-account-key.json"


from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
import clip


# --- CONFIGURATION ---
# UPDATE THESE PATHS TO YOUR LOCAL PATHS
TEXT_MODEL_NAME = "t5-base"  # Or your specific fine-tuned path
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CLIP_MODEL_NAME = "ViT-B/32"
CLIP_WEIGHTS_PATH = "CLIP__best_model.pt" # Update this
TEXT_WEIGHTS_PATH = "T5__best_model.pt" # Update this

MAX_LENGTH = 128
SENTIMENTS_TO_LABELS = {'negative':0,'neutral':1,'positive':2}
LABELS_TO_SENTIMENTS = {0: 'Negative', 1: 'Neutral', 2: "Positive"}

csv_filepath = "./docimsentv1.csv"
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

    # 2. Load Text Model (T5)
    # Note: Ensure num_labels matches your training (e.g., 3 for Neg/Neu/Pos)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, cache_dir=cache_dir)
    text_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME, num_labels=3, cache_dir=cache_dir)
    
    # Load weights if they exist, otherwise use base (warning: base model won't predict sentiment correctly without fine-tuning)
    if os.path.exists(TEXT_WEIGHTS_PATH):
        text_model.load_state_dict(torch.load(TEXT_WEIGHTS_PATH, map_location=device))
    text_model.to(device)
    text_model.eval()

    # 3. Load CLIP (Visual)
    # clip_path = os.path.join(cache_dir, "ViT-B-32.pt")
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
        # return texts[0].description, texts[0]
        # above is used when adaptive-alpha is used;
        return texts[0].description
    return None
 

# Way-1 when using the adaptive-alpha in your logic
def check_text_detection_v1(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    client = vision.ImageAnnotatorClient()
    
    image = vision.Image(content=content)
    
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0] if texts else False


# Way-2 when using the adaptive-alpha in your logic
def check_text_detection_v2(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    client = vision.ImageAnnotatorClient()
    
    image = vision.Image(content=content)
    text = None
    
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts if texts else False


# get_alpha_v0 calculates text area as CLIPPED union of all the text-areas
def get_alpha_v0(ocr_annotation, img_width, img_height):
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


# get_alpha_v1 calculates text area as MINIMUM union of all the text-areas
def get_alpha_v1(texts, img_area):
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
    

# Shoelace formula
def polygon_area(vertices):
    """
    Calculate the area of a polygon given its vertices.
    vertices: list of (x, y) tuples
    Returns: absolute area of the polygon
    """
    n = len(vertices)
    if n<3:
        return 0  # Not a polygon
    area = 0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i+1)%n]
        area += x1*y2 - x2*y1
    return abs(area)/2


# get_alpha_v2 calculates text area as MAXIMUM union of all the text-areas
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


def get_from_BLIP(models_dict, image_pil, device):
    # Fault Tolerance component: Generate caption via BLIP
    inputs = models_dict["blip_processor"](images=image_pil, return_tensors="pt").to(device)
    out = models_dict["blip_model"].generate(**inputs)
    return models_dict["blip_processor"].decode(out[0], skip_special_tokens=True)


# --- MAIN INFERENCE PIPELINE (MeVITSA) ---
def mevitsa_analysis(image_pil, image_bytes, models_dict, manual_alpha=None, gcv_api=1, ft=1):
    """
    The MeVITSA Method:
    1. Extract Text (OCR from CSV) -> if failed, Generate Caption (BLIP).
    2. Get Text Sentiment (T5).
    3. Get Image Sentiment (CLIP).
    4. Fuse using Alpha (Area-based or Manual or Fixed).
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
    
    if gcv_api==1:
        # ocr_text, ocr_annotation = get_ocr_text(image_bytes) 
        # ocr_annotation is used when adaptive alpha is used; 
        ocr_text = get_ocr_text(image_bytes)
        source = "OCR"
        if ocr_text is not None and ocr_text != "":
            final_text = ocr_text
        else:
            if ft==1:
                final_text = get_from_BLIP(models_dict, image_pil, device)
                source = "BLIP"
            else:
                final_text = ""
    else:
        if ft==1:
            final_text = get_from_BLIP(models_dict, image_pil, device)
            source = "BLIP"
        else:
            final_text = ""
    
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

    # 3. FUSION (MeVITSA LOGIC)
    # ----------------------
    if manual_alpha is not None:
        alpha = manual_alpha
    else:
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


# ---------------- CONFIGURATION ----------------
# Path to your image folder
DATASET_FOLDER = "./dataset" 

# Ensure models are loaded
# if 'models' not in locals():
print("Loading models...")
models = load_core_models()

# ---------------- HELPER: PARSE LABEL ----------------
def get_true_label(filename):
    """
    Parses sentiment from filename.
    Assumes format like: 'positive_123.jpg' or '123_negative.jpg'
    """
    filename_lower = filename.lower()
    if "positive" in filename_lower:
        return "Positive"
    elif "negative" in filename_lower:
        return "Negative"
    elif "neutral" in filename_lower:
        return "Neutral"
    return None

# ---------------- MAIN ANALYSIS LOOP ----------------
results = []
valid_extensions = ('.jpg', '.jpeg', '.png', ".JPG")
image_files = [f for f in os.listdir(DATASET_FOLDER) if f.lower().endswith(valid_extensions)]

print(f"Found {len(image_files)} images. Starting analysis...")

for img_name in tqdm(image_files):
    img_path = os.path.join(DATASET_FOLDER, img_name)
    
    # 1. Get Ground Truth
    true_label = get_true_label(img_name)
    if not true_label:
        continue # Skip if no label found in filename
        
    # 2. Prepare Image
    try:
        image_pil = Image.open(img_path).convert("RGB")
        # specific to your pipeline: needs bytes for OCR check (if using API)
        # or filename lookup if relying on CSV
        with open(img_path, "rb") as f:
            image_bytes = f.read()
            
        # 3. Run MeViTSA Inference
        prediction = mevitsa_analysis( 
            image_pil=image_pil, 
            image_bytes=image_bytes, 
            models_dict=models,
            gcv_api=0
        )
        
        # 4. Log Data
        results.append({
            "filename": img_name,
            "true_label": true_label,
            "pred_label": prediction['final_class'],
            "confidence": max(prediction['final_probs']),
            "text_source": prediction['text_source'],
            "alpha": prediction['alpha_used'],
            "extracted_text": prediction['text_content'] # [:50] + "..." # You may truncate for display
        })
        
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
    # break



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- METRICS & VISUALIZATION ----------------
df_results = pd.DataFrame(results)
# Save the DataFrame to a CSV file
df_results.to_csv("evaluation_results.csv", index=False)
print("Predictions saved to: evaluation_results.csv")

# 1. Basic Accuracy
acc = accuracy_score(df_results['true_label'], df_results['pred_label'])
print()
print(f"\nOverall Accuracy: {acc:.4%}")

# 2. Classification Report
print()
print("Classification Report:")
report = classification_report(df_results['true_label'], df_results['pred_label'], digits=4)
print(report)

with open("classification_report.txt", "w") as f:
    f.write("MeViTSA Performance Report\n")
    f.write("==========================\n")
    f.write(f"Overall Accuracy: {acc:.4%}\n\n")
    f.write(report)

print("Report saved to: classification_report.txt")

# 3. Confusion Matrix
plt.figure(figsize=(8, 6))
labels = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(df_results['true_label'], df_results['pred_label'], labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title(f"MeViTSA confusion matrix (Accuracy: {acc:.4f})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()