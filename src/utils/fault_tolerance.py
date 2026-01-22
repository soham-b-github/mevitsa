import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_blip(model_name, device, cache_dir):
    processor = BlipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)
    return model, processor

def generate_fault_tolerant_caption(model, processor, image_pil, device):
    inputs = processor(images=image_pil, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)