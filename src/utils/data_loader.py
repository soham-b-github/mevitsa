import os
import pandas as pd

class DatasetHandler:
    def __init__(self, dataset_path, csv_name="docimsentv1.csv"):
        self.dataset_path = dataset_path
        self.csv_path = os.path.join(dataset_path, csv_name)
        
    def get_image_files(self, extensions=('.jpg', '.jpeg', '.png', '.JPG')):
        """Retrieves all valid image files from the dataset folder."""
        return [f for f in os.listdir(self.dataset_path) 
                if f.lower().endswith(extensions)]

    @staticmethod
    def get_true_label(filename):
        """Parses sentiment from filename as per model logic."""
        fn = filename.lower()
        if "positive" in fn: return "Positive"
        if "negative" in fn: return "Negative"
        if "neutral" in fn: return "Neutral"
        return None

    def load_metadata(self):
        """Loads the accompanying CSV if it exists."""
        if os.path.exists(self.csv_path):
            return pd.read_csv(self.csv_path)
        return None