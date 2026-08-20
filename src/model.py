import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models.v_br import load_vision_branch
from models.t_br import load_text_branch
from utils.fault_tolerance import load_blip, generate_fault_tolerant_caption
from utils.ocr_engine import get_ocr_text
from utils.data_loader import DatasetHandler

class MeViTSA:
    def __init__(self, config):
        # ALWAYS load models onto CPU during app startup
        self.device = "cpu"
        self.config = config
        self.models_on_gpu = False
        
        # Load vision and text branches
        self.v_model, self.v_preprocess = load_vision_branch(
            config['vision_model']['name'], config['vision_model']['weights_path'], 
            config['num_classes'], self.device, config['cache_dir']
        )
        self.t_model, self.t_tokenizer = load_text_branch(
            config['text_model']['name'], config['text_model']['weights_path'], 
            config['num_classes'], self.device, config['cache_dir']
        )
        self.blip_model, self.blip_proc = load_blip(
            config['caption_model']['name'], self.device, config['cache_dir']
        )
        
        # CSV-based OCR lookup initialization
        self.dh = DatasetHandler(config.get('dataset_path', "./"))
        self.df = self.dh.load_metadata()

    def _ensure_gpu(self):
        """Move models to GPU on-demand inside the @spaces.GPU context."""
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        if not self.models_on_gpu and target_device == "cuda":
            self.v_model = self.v_model.to(target_device)
            self.t_model = self.t_model.to(target_device)
            self.blip_model = self.blip_model.to(target_device)
            self.device = target_device
            self.models_on_gpu = True

    def analyze(self, image_pil, image_bytes, image_filename="", manual_alpha=None, gcv_api=True, ft=True):
        # Dynamically shift models to CUDA now that @spaces.GPU is active
        self._ensure_gpu()
        
        # 1. Visual Pipeline (CLIP)
        img_tensor = self.v_preprocess(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v_probs = F.softmax(self.v_model(img_tensor), dim=1).cpu().numpy().flatten()
        
        # 2. Text Pipeline Logic
        final_text = ""
        source = "None"

        if gcv_api:
            ocr_text = get_ocr_text(image_bytes)
            if ocr_text:
                final_text = ocr_text
                source = "OCR (GCV)"
        else:
            # Fallback to CSV
            print("Is self.df == None?", (self.df==None))
            print("image_filename:", image_filename)
            
            if self.df is not None and image_filename in self.df["filename"].values:
                matched_row = self.df.loc[self.df["filename"] == image_filename].iloc[0]
                ocr_text = matched_row.get("OCR", "")
                if pd.notna(ocr_text) and str(ocr_text).strip() != "":
                    final_text = str(ocr_text)
                    source = "OCR (CSV)"

        # 3. Fault Tolerance (BLIP)
        if not final_text and ft:
            final_text = generate_fault_tolerant_caption(self.blip_model, self.blip_proc, image_pil, self.device)
            source = "BLIP (FT)"

        # 4. Text Inference (T5)
        if final_text:
            t_inputs = self.t_tokenizer(
                final_text, return_tensors="pt", truncation=True, padding=True, 
                max_length=self.config['text_model'].get('max_length', 128)
            ).to(self.device)
            with torch.no_grad():
                t_probs = F.softmax(self.t_model(**t_inputs).logits, dim=1).cpu().numpy().flatten()
        else:
            t_probs = np.array([0.0, 0.0, 0.0])

        # 5. MeViTSA Weighted Fusion
        alpha = manual_alpha if manual_alpha is not None else 0.5
        final_probs = (alpha * t_probs) + ((1 - alpha) * v_probs)
        print(t_probs)
        print(v_probs)
        
        return {
            "label": self.config['labels'][str(np.argmax(final_probs))],
            "probs": final_probs,
            "text": final_text,
            "source": source,
            "alpha": alpha
        }
