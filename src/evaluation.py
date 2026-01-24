import os
import pandas as pd
import argparse
import torch
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from model import MeViTSA # type: ignore
from utils.data_loader import DatasetHandler

def run_evaluation(dataset_path):
    # 1. Setup
    with open("./../configs/config-models.json") as f:
        config = json.load(f)
    
    engine = MeViTSA(config)
    dh = DatasetHandler(dataset_path)
    image_files = dh.get_image_files()
    
    results = []
    print(f"Found {len(image_files)} images. Starting batch analysis...")

    # 2. Analysis Loop
    for img_name in tqdm(image_files):
        img_path = os.path.join(dataset_path, img_name)
        true_label = dh.get_true_label(img_name)
        
        if not true_label:
            continue
            
        try:
            from PIL import Image
            image_pil = Image.open(img_path).convert("RGB")
            with open(img_path, "rb") as f:
                image_bytes = f.read()
                
            prediction = engine.analyze(image_pil, image_bytes)
            
            results.append({
                "filename": img_name,
                "true_label": true_label,
                "pred_label": prediction['label'],
                "confidence": max(prediction['probs']),
                "text_source": prediction['source'],
                "alpha": prediction['alpha'],
                "extracted_text": prediction['text']
            })
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    # 3. Metrics Generation
    df_results = pd.DataFrame(results)
    df_results.to_csv("evaluation_results.csv", index=False)
    
    acc = accuracy_score(df_results['true_label'], df_results['pred_label'])
    report = classification_report(df_results['true_label'], df_results['pred_label'], digits=4)
    
    print(f"\nOverall Accuracy: {acc:.4%}")
    print("\nClassification Report:\n", report)

    # 4. Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(df_results['true_label'], df_results['pred_label'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"MeViTSA Confusion Matrix (Acc: {acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a specific dataset.")
    
    # Adding the command line argument
    parser.add_argument(
        "--DATASET_PATH", 
        type=str, 
        default="./../data/docimsentv1-samples", 
        help="Path to the dataset folder (e.g., ./../data/your_dataset)"
    )
    
    args = parser.parse_args()
    
    # Access the path via args.DATASET_PATH
    print(f"Starting evaluation using dataset at: {args.DATASET_PATH}")
    run_evaluation(args.DATASET_PATH)