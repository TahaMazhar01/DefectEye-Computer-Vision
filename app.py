"""Gradio demo for DefectEye — deployable to Hugging Face Spaces.

Drop in a product image -> get a defect heatmap, an anomaly score, and a
PASS / FAIL verdict. Loads any trained model found in ./models.

Run locally:   python app.py
On HF Spaces:  this file is the entry point (keep it named app.py at repo root).
"""

import glob
import os

import gradio as gr
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.padim import PaDiM
from src.utils import make_overlay

MODELS_DIR = "models"
CROPSIZE = 224

_transform = T.Compose([
    T.Resize(256, T.InterpolationMode.BILINEAR),
    T.CenterCrop(CROPSIZE),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# discover trained models: filename pattern is <category>_<backbone>.pt
_model_paths = {os.path.splitext(os.path.basename(p))[0]: p
                for p in sorted(glob.glob(os.path.join(MODELS_DIR, "*.pt")))}
_cache = {}


def _load(name):
    if name not in _cache:
        path = _model_paths[name]
        backbone = "wide_resnet50_2" if "wide_resnet50_2" in name else "resnet18"
        model = PaDiM(backbone=backbone, device="cpu")
        model.load_state(torch.load(path, map_location="cpu"))
        _cache[name] = model
    return _cache[name]


def analyze(image, model_name):
    if image is None:
        return None, "Upload an image first."
    model = _load(model_name)
    x = _transform(image.convert("RGB")).unsqueeze(0)

    from src.utils import upsample_map
    amap = upsample_map(model.predict(x), CROPSIZE)[0]

    score = float(amap.max())
    rng = model.score_max - model.score_min + 1e-8
    norm_score = (score - model.score_min) / rng
    is_defect = score > model.threshold

    # normalize the heatmap for display using the calibrated range
    heat01 = np.clip((amap - model.score_min) / rng, 0, 1)
    base = np.asarray(image.convert("RGB").resize((CROPSIZE, CROPSIZE))) / 255.0
    overlay_bgr = make_overlay(base, heat01)
    overlay_rgb = overlay_bgr[..., ::-1]

    verdict = "❌ DEFECT DETECTED" if is_defect else "✅ PASS — looks normal"
    label = (f"{verdict}\n\nAnomaly score: {norm_score:.3f}  "
             f"(threshold: {(model.threshold - model.score_min) / rng:.3f})")
    return overlay_rgb, label


with gr.Blocks(title="DefectEye") as demo:
    gr.Markdown(
        "# 🔍 DefectEye — Industrial Anomaly Detection\n"
        "Unsupervised defect detection with **PaDiM**. The model was trained on "
        "*defect-free* images only, yet localizes unseen defects as a heatmap.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Product image")
            model_dd = gr.Dropdown(
                choices=list(_model_paths.keys()),
                value=(list(_model_paths.keys())[0] if _model_paths else None),
                label="Trained model")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Defect heatmap")
            out_txt = gr.Textbox(label="Verdict", lines=3)
    btn.click(analyze, inputs=[inp, model_dd], outputs=[out_img, out_txt])

if __name__ == "__main__":
    if not _model_paths:
        print("WARNING: no models found in ./models — train one first "
              "(python -m src.train ...) or download a trained .pt.")
    demo.launch()
