import torch
import torch.nn as nn
import clip

class CLIPClassifier(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, image):
        image = image.to(dtype=self.encoder.conv1.weight.dtype)
        with torch.no_grad():
            x = self.encoder(image)
        return self.classifier(x.float())

def load_vision_branch(model_name, weights_path, num_classes, device, cache_dir):
    clip_base, preprocess = clip.load(model_name, device=device, download_root=cache_dir)
    embed_dim = clip_base.visual.output_dim 
    model = CLIPClassifier(clip_base.visual, embed_dim, num_classes).to(device)
    
    import os
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model, preprocess