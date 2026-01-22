import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_text_branch(model_name, weights_path, num_classes, device, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes, cache_dir=cache_dir
    )
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer