# MeViTSA: Multimodal Ensemble for Visuals Integrated Text Sentiment Analysis

**Term Project** | M.Sc. Big Data Analytics  
**Institution:** Ramakrishna Mission Vivekananda Educational and Research Institute, Belur  
**Author:** Soham Bhattacharya (ID: B2430059)  
**Supervisors:** Prof. Umapada Pal & Prof. Ali Reza Alaei

---

## 📌 Project Overview
**MeViTSA** (Multimodal Ensemble for Visuals Integrated Text Sentiment Analysis) is a deep learning framework designed to classify sentiment in images containing embedded text—such as memes, advertisements, and quote cards.

Unlike traditional multimodal models that treat text and images as separate files, MeViTSA is optimized for **visually integrated text**. It achieves state-of-the-art performance (96.59% accuracy) on the curated **DocImSentv1** dataset.

### Key Features
* **Dual-Stream Architecture:** Processes visual and textual modalities independently before fusion.
* **Textual Channel:** Uses **Google Cloud Vision API** for robust OCR and **T5** for sentiment encoding.
* **Visual Channel:** Utilizes **CLIP (ViT-B/32)** to extract high-dimensional semantic visual embeddings.
* **Fault Tolerance (FT):** A novel mechanism using **BLIP** to generate descriptive captions when OCR fails, ensuring no input is discarded.
* **Late Fusion:** Aggregates predictions using a static weighted strategy for maximum stability. 

---

## ⚠️ Important Usage Note: Trained Models Required

The file `app.py` included in this repository serves as the **frontend interface** for the inference engine. 

> **🛑 CRITICAL:** > This application **will not function** without the pre-trained model weights (e.g., `best_model.pth` or checkpoint files). Due to file size limitations, these model files are **not included** in this repository.

**How to get the models:**
Please **contact the author** directly to request access to the trained model files. Once received, place them in the root directory (or the specified `models/` folder) before running the application.

---

## 🛠️ Architecture

The framework consists of two parallel pipelines:
1.  **Text Pipeline:** * Input Image $\rightarrow$ OCR (Google Cloud Vision) $\rightarrow$ Text Preprocessing $\rightarrow$ **T5 Encoder**.
    * *Fallback:* If OCR $\rightarrow$ Empty $\rightarrow$ **BLIP Captioning** $\rightarrow$ T5 Encoder.
2.  **Visual Pipeline:**
    * Input Image $\rightarrow$ Preprocessing $\rightarrow$ **CLIP Image Encoder**. 
3.  **Fusion:**
    * Outputs are combined via Late Fusion: $P_{final} = \alpha \cdot P_{text} + (1-\alpha) \cdot P_{vis}$

---

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/soham-b-github/msa.git](https://github.com/soham-b-github/msa.git)
    cd msa
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Key requirements:* `torch`, `transformers`, `google-cloud-vision`, `clip` (OpenAI), `Pillow`, `flask` / `streamlit` (for app.py).

3.  **Setup Google Cloud Credentials**
    * Export your service account key:
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"
        ```

---

## 🚀 Running the App

Once you have obtained the trained model files from the author:

1.  Ensure the model file is in the correct directory.
2.  Run the frontend application:
    ```bash
    streamlit run app.py
    ```
3.  Upload an image (e.g., a meme or poster) to receive the predicted sentiment (Positive, Negative, or Neutral).

---

## 📊 Performance

| Dataset | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **DocImSentv1** | **96.59%** | 0.9657 |
| MVSA-Single | 79.11% | 0.716 |
| MVSA-Multiple | 73.99% | 0.655 |


---

## 📧 Contact

For model files, collaboration, or queries regarding the **DocImSentv1** dataset, please contact:

**Soham Bhattacharya** M.Sc. Big Data Analytics  
RKMVERI, Belur
