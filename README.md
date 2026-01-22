# MeViTSA: Multimodal Ensemble Approach for Sentiment Analysis of Visuals Integrated Text Data

**Authors:** Soham Bhattacharya<sup>1,3</sup>, Ali Reza Alaei<sup>2</sup> & Umapada Pal<sup>3</sup>

**Affiliated institutions:** <sup>1</sup>Ramakrishna Mission Vivekananda Educational and Research Institute Belur, India; <sup>2</sup>Southern Cross University, Australia; <sup>3</sup>Indian Statistical Institute Kolkata, India;


### Primary contributions
* The development of an enhanced MSA framework that demonstrates superior performance over the existing baselines
* The integration of a novel Fault-Tolerance (FT) mechanism that ensures architectural robustness and handles out-of-distribution (OOD) data which are faulty inputs
* Extending and a comprehensive curation and refinement of the dataset DocImsent introduced by Ahuja et. al (2024) to improve data quality and volume of the dataset.

## Architecture

![framework-arch](assets/framework.png)

## Important note about usage: Trained models required

The file `app.py` included in this repository serves as the **frontend interface** for the inference engine. 

> **CRITICAL:** > This application **will not function** without the pre-trained model weights (e.g., `best_model.pth` or checkpoint files). Due to file size limitations, these model files are **not included** in this repository.

**How to get the models:**
Please **contact the author** directly to request access to the trained model files. Once received, place them in the root directory (or the specified `models/` folder) before running the application.


## Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/soham-b-github/msa.git](https://github.com/soham-b-github/msa.git)
    cd msa
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *Key requirements:* `torch`, `transformers`, `google-cloud-vision`, `clip` (OpenAI), `Pillow`, `flask` / `streamlit` (for app.py).

3.  **Setup Google Cloud credentials**
    * Export your service account key:
        ```bash
        export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"
        ```


## Running the App

Once you have obtained the trained model files from the author:

1.  Ensure the model file is in the correct directory.
2.  Run the frontend application:
    ```bash
    streamlit run app.py
    ```
3.  Upload an image (e.g., a meme or poster) to receive the predicted sentiment (Positive, Negative, or Neutral).
