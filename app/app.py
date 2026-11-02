"""Drishti - explainable disaster detection from drone images (Gradio app / Hugging Face Space entry point).

Run locally:   python app.py        ->  http://127.0.0.1:7860
"""
import os
from pathlib import Path

try:
    import spaces          # Hugging Face ZeroGPU support (imported before torch)
except ImportError:        # running locally without the `spaces` package: make @spaces.GPU a no-op
    class spaces:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            return fn if callable(fn) else (lambda f: f)

import gradio as gr
from PIL import ImageOps

from drishti_core import (PLAYBOOK, DrishtiEngine, alert_card_html, draw_scan, overlay_heatmap,
                          scan_report_markdown)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("DRISHTI_MODEL", APP_DIR / "model" / "drishti_model.pt"))
EXAMPLE_DIR = APP_DIR / "examples"
EXAMPLES = sorted(str(p) for p in EXAMPLE_DIR.glob("*.jpg") if not p.name.startswith("area_scan"))
SCAN_EXAMPLES = sorted(str(p) for p in EXAMPLE_DIR.glob("area_scan*.jpg"))
EXAMPLE_NAMES = {"collapsed_building": "Collapsed building", "fire": "Fire / smoke", "flooded_areas": "Flood",
                 "normal": "Normal", "traffic_incident": "Traffic accident"}


def _example_label(path):
    stem = Path(path).stem               # e.g. "flooded_areas_2"
    cls = stem.rsplit("_", 1)[0]
    return f"{EXAMPLE_NAMES.get(cls, cls)} #{stem.rsplit('_', 1)[-1]}"

try:
    ENGINE, LOAD_ERROR = DrishtiEngine(MODEL_PATH), None
except Exception as err:   # keep the UI alive and show the problem instead of crashing the Space
    ENGINE, LOAD_ERROR = None, f"Model could not be loaded from '{MODEL_PATH}': {err}"
    print(LOAD_ERROR)


def _prepare(img):
    if ENGINE is None:
        raise gr.Error(LOAD_ERROR)
    if img is None:
        raise gr.Error("Please upload an aerial / drone image first.")
    return ImageOps.exif_transpose(img).convert("RGB")


# Only the model calls run on the GPU (ZeroGPU attaches a GPU just for the duration of these functions).
@spaces.GPU(duration=20)
def _classify_on_gpu(img, explain):
    if explain:
        return ENGINE.gradcam(img)            # (cam, probs)
    return None, ENGINE.predict(img)


@spaces.GPU(duration=20)
def _scan_on_gpu(img, grid):
    return ENGINE.scan(img, grid)             # (boxes, probs)


def analyze(img, explain):
    img = _prepare(img)
    cam, probs = _classify_on_gpu(img, explain)
    heatmap = overlay_heatmap(img, cam) if cam is not None else None
    scores = {ENGINE.display[c]: float(p) for c, p in zip(ENGINE.classes, probs)}
    return alert_card_html(ENGINE, probs), scores, heatmap


def area_scan(img, grid):
    img = _prepare(img)
    boxes, probs = _scan_on_gpu(img, int(grid))
    return draw_scan(ENGINE, img, boxes, probs), scan_report_markdown(ENGINE, probs)


def about_markdown():
    if ENGINE is None:
        return f"**Model not loaded.** {LOAD_ERROR}"
    m = ENGINE.metrics
    test, calib = m.get("test", {}), m.get("calibration", {})
    recall = m.get("per_class_recall", {})
    recall_rows = "\n".join(f"| {ENGINE.display.get(c, c)} | {recall[c]:.1%} | {PLAYBOOK[c]['severity']} |"
                            for c in ENGINE.classes if c in recall)
    return f"""
### Model card
| | |
|---|---|
| Backbone | `{ENGINE.arch}` (ImageNet-pretrained, fine-tuned) |
| Classes | {", ".join(ENGINE.display[c] for c in ENGINE.classes)} |
| Test accuracy | {test.get("acc", float("nan")):.1%} |
| Test macro-F1 | {test.get("macro_f1", float("nan")):.3f} |
| Calibration | temperature T = {calib.get("temperature", 1.0):.2f}, ECE {calib.get("ece_before", float("nan")):.3f} → {calib.get("ece_after", float("nan")):.3f} |
| Human-in-the-loop | at {ENGINE.threshold:.0%} confidence, {calib.get("auto_handled_share", float("nan")):.0%} of test images are handled automatically with {calib.get("accuracy_auto_handled", float("nan")):.1%} accuracy |
| Trained | {ENGINE.created} |

| Class | Test recall | Severity |
|---|---|---|
{recall_rows}

### How it works
1. The image is resized to {ENGINE.img_size}×{ENGINE.img_size} and classified by a CNN fine-tuned on the AIDER dataset.
2. Probabilities are **calibrated** with temperature scaling, so "80 %" really means right about 8 times in 10.
3. **Grad-CAM** highlights the regions that drove the decision.
4. **Area scan** cuts a large frame into tiles and classifies each one to map *where* the emergency is.

### Limitations
* AIDER images were collected from the web; real drone footage in India (different terrain, seasons, camera angles) can look different.
* Scene-level classifier: it says *what* is in an image or tile, not exact object boundaries.
* A decision-support tool: low-confidence frames are deliberately routed to a human.

### Credits
Dataset: AIDER — C. Kyrkou & T. Theocharides, *EmergencyNet*, IEEE JSTARS 13 (2020) 1687–1699; Zenodo DOI 10.5281/zenodo.3888300 (CC BY 4.0).
Built for EC-422 Artificial Intelligence, IIT (BHU) Varanasi.
"""


HEADER = """
# 🛰️ Drishti — explainable disaster detection from drone images
Upload an aerial or drone photo. The model labels it as **collapsed building, fire / smoke, flood, traffic accident or normal**,
shows **where** it looked (Grad-CAM), suggests a **response**, and hands **uncertain** frames to a human.
"""

with gr.Blocks(title="Drishti - disaster detection from drone images") as demo:
    gr.Markdown(HEADER)
    if LOAD_ERROR:
        gr.Markdown(f"> ⚠️ **{LOAD_ERROR}** — copy `drishti_model.pt` from the Kaggle outputs into `app/model/`.")

    with gr.Tab("🔍 Analyze an image"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Aerial / drone image", sources=["upload", "clipboard"],
                                    height=360)
                explain_in = gr.Checkbox(value=True, label="Explain the decision with a Grad-CAM heatmap")
                analyze_btn = gr.Button("Analyze", variant="primary")
                if EXAMPLES:
                    gr.Examples(examples=[[p] for p in EXAMPLES], inputs=[image_in],
                                example_labels=[_example_label(p) for p in EXAMPLES],
                                label="Examples from the held-out test set", examples_per_page=10)
            with gr.Column():
                alert_out = gr.HTML()
                label_out = gr.Label(num_top_classes=len(ENGINE.classes) if ENGINE else 5,
                                     label="Calibrated class probabilities")
                cam_out = gr.Image(type="pil", label="Grad-CAM — red = regions that drove the decision")
        analyze_btn.click(analyze, inputs=[image_in, explain_in], outputs=[alert_out, label_out, cam_out])

    with gr.Tab("🗺️ Area scan (large drone frame)"):
        gr.Markdown("Cuts a large frame into an N × N grid and classifies every tile to show **where** "
                    "the emergency is. Coloured tiles = emergency, outlined only = normal, '?' = low confidence.")
        with gr.Row():
            with gr.Column():
                scan_in = gr.Image(type="pil", label="Large aerial image", sources=["upload", "clipboard"],
                                   height=360)
                grid_in = gr.Slider(2, 8, value=4, step=1, label="Grid size (N × N tiles)")
                scan_btn = gr.Button("Scan area", variant="primary")
                if SCAN_EXAMPLES:
                    gr.Examples(examples=[[p, 4] for p in SCAN_EXAMPLES], inputs=[scan_in, grid_in],
                                label="Example: four test photos stitched into one frame")
            with gr.Column():
                scan_out = gr.Image(type="pil", label="Disaster map")
                report_out = gr.Markdown()
        scan_btn.click(area_scan, inputs=[scan_in, grid_in], outputs=[scan_out, report_out])

    with gr.Tab("ℹ️ About the model"):
        gr.Markdown(about_markdown())

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"))
