import io
from google.cloud import vision

def get_ocr_text(image_content):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else None

def calculate_alpha(ocr_annotation, img_width, img_height):
    if not ocr_annotation:
        return 0.5
    # logic from get_alpha_v0
    vertices = ocr_annotation.bounding_poly.vertices
    x_vals = [v.x for v in vertices]
    y_vals = [v.y for v in vertices]
    txt_area = (max(x_vals) - min(x_vals)) * (max(y_vals) - min(y_vals))
    alpha = txt_area / (img_width * img_height)
    return min(max(alpha, 0.2), 0.8)