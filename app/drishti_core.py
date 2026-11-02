"""Drishti inference engine: model loading, calibrated prediction, Grad-CAM and area scan.

Kept free of Gradio so it can be reused from scripts and tests:

    from PIL import Image
    from drishti_core import DrishtiEngine
    engine = DrishtiEngine("model/drishti_model.pt")
    probs = engine.predict(Image.open("drone_photo.jpg"))
"""
from __future__ import annotations

import html
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms as T

DEFAULT_COLORS = {"collapsed_building": "#8d6e63", "fire": "#e4572e", "flooded_areas": "#2e86de",
                  "normal": "#43a047", "traffic_incident": "#f2a900"}

# What a prediction means for an emergency operator in India
# (ERSS 112 = national emergency number, 101 = fire services, 108 = ambulance in most states).
PLAYBOOK = {
    "collapsed_building": dict(
        severity="CRITICAL", rank=1, color="#b71c1c",
        action="Dispatch search-and-rescue (NDRF / SDRF) and ambulances (108); request a structural "
               "assessment before teams enter; report through ERSS 112."),
    "fire": dict(
        severity="CRITICAL", rank=1, color="#d84315",
        action="Alert Fire & Emergency Services (101); check wind direction and nearby houses, "
               "forests or fuel stations; prepare evacuation."),
    "flooded_areas": dict(
        severity="HIGH", rank=2, color="#1565c0",
        action="Notify the District Disaster Management Authority; plan boat rescue, safe evacuation "
               "routes and relief camps."),
    "traffic_incident": dict(
        severity="HIGH", rank=2, color="#b26a00",
        action="Send police and an ambulance (112 / 108); divert traffic and secure the site."),
    "normal": dict(
        severity="NONE", rank=4, color="#2e7d32",
        action="No emergency visible in this frame - continue routine patrol."),
}
REVIEW_COLOR = "#5e35b1"


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Same architecture as in the training notebook (torchvision backbone + new output layer)."""
    if not arch.startswith(("efficientnet", "mobilenet_v3")):
        raise ValueError(f"Unsupported architecture '{arch}'")
    model = torchvision.models.get_model(arch, weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def tile_boxes(width: int, height: int, grid: int):
    return [(c * width // grid, r * height // grid, (c + 1) * width // grid, (r + 1) * height // grid)
            for r in range(grid) for c in range(grid)]


class DrishtiEngine:
    """Loads drishti_model.pt (written by the Kaggle notebook) and runs all inference on CPU."""

    def __init__(self, checkpoint_path: str | Path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.arch = ckpt["arch"]
        self.classes = list(ckpt["classes"])
        self.display = dict(ckpt["display_names"])
        self.colors = {**DEFAULT_COLORS, **dict(ckpt.get("class_colors", {}))}
        self.temperature = float(ckpt.get("temperature", 1.0))
        self.threshold = float(ckpt.get("uncertainty_threshold", 0.6))
        self.cache_size = int(ckpt.get("cache_size", 256))
        self.img_size = int(ckpt.get("img_size", 224))
        self.metrics = dict(ckpt.get("metrics", {}))
        self.created = ckpt.get("created", "")
        self.model = build_model(self.arch, len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        # GPU if available. On a ZeroGPU Space the model must be put on "cuda" at startup (it is emulated
        # until a @spaces.GPU function runs); locally without a GPU everything stays on the CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.cam_layer = self.model.features[-1]
        self.transform = T.Compose([T.Resize((self.img_size, self.img_size)), T.ToTensor(),
                                    T.Normalize(ckpt["mean"], ckpt["std"])])
        self._lock = threading.Lock()   # Grad-CAM attaches hooks; never run two requests on the model at once

    # ---------------------------------------------------------------- helpers
    def _tensor(self, img: Image.Image) -> torch.Tensor:
        # identical to the notebook: square resize to the cache size, then to the network size
        img = img.convert("RGB").resize((self.cache_size, self.cache_size), Image.Resampling.BILINEAR)
        return self.transform(img)

    def _calibrated(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits.float() / self.temperature).softmax(-1)

    # ---------------------------------------------------------------- inference
    def predict(self, img: Image.Image) -> np.ndarray:
        """Calibrated class probabilities, shape (num_classes,)."""
        with self._lock, torch.inference_mode():
            x = self._tensor(img).unsqueeze(0).to(self.device)
            return self._calibrated(self.model(x))[0].cpu().numpy()

    def gradcam(self, img: Image.Image, class_idx: int | None = None):
        """Returns (cam, probs): cam is a low-resolution map in [0, 1] for the predicted (or given) class."""
        x = self._tensor(img).unsqueeze(0).to(self.device).requires_grad_(True)
        store = {}

        def hook(_module, _inputs, out):
            store["act"] = out
            out.register_hook(lambda g: store.__setitem__("grad", g))

        with self._lock:
            handle = self.cam_layer.register_forward_hook(hook)
            try:
                with torch.enable_grad():
                    self.model.zero_grad(set_to_none=True)
                    logits = self.model(x)
                    idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
                    logits[0, idx].backward()
            finally:
                handle.remove()
                self.model.zero_grad(set_to_none=True)
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * store["act"]).sum(1))[0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), self._calibrated(logits.detach())[0].cpu().numpy()

    def scan(self, img: Image.Image, grid: int = 4):
        """Area scan: classify grid x grid tiles in one batch. Returns (boxes, probs[n_tiles, n_classes])."""
        img = img.convert("RGB")
        boxes = tile_boxes(*img.size, grid)
        with self._lock, torch.inference_mode():
            x = torch.stack([self._tensor(img.crop(b)) for b in boxes]).to(self.device)
            probs = self._calibrated(self.model(x)).cpu().numpy()
        return boxes, probs


# -------------------------------------------------------------------- drawing
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _rgb(hex_color: str):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.45, max_side: int = 768) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    heat = Image.fromarray(np.uint8(np.clip(cam, 0, 1) * 255)).resize(img.size, Image.Resampling.BICUBIC)
    heat = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    out = (1 - alpha) * np.asarray(img) / 255.0 + alpha * heat
    return Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255))


def draw_scan(engine: DrishtiEngine, img: Image.Image, boxes, probs, max_side: int = 1024) -> Image.Image:
    """Colour every tile by its predicted class; emergency tiles are filled, normal tiles only outlined."""
    img = img.convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.BILINEAR)
        boxes = [tuple(round(v * scale) for v in b) for b in boxes]
    base = img.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    tile_w = max(1, boxes[0][2] - boxes[0][0])
    longest = max((engine.display.get(c, c) for c in engine.classes), key=len) + " 100% ?"
    size = max(8, min(22, tile_w // 8))
    font = _font(size)
    while size > 8 and draw.textlength(longest, font=font) > tile_w - 14:   # shrink until labels fit a tile
        size -= 1
        font = _font(size)
    for box, p in zip(boxes, probs):
        k = int(np.argmax(p))
        cls = engine.classes[k]
        rgb = _rgb(engine.colors.get(cls, "#888888"))
        uncertain = p[k] < engine.threshold
        if cls != "normal" and not uncertain:
            draw.rectangle(box, fill=rgb + (int(60 + 110 * p[k]),))
        draw.rectangle(box, outline=rgb + (255,), width=3 if not uncertain else 1)
        text = f"{engine.display.get(cls, cls)} {p[k]:.0%}" + (" ?" if uncertain else "")
        tb = draw.textbbox((box[0] + 6, box[1] + 5), text, font=font)
        draw.rectangle((tb[0] - 3, tb[1] - 2, tb[2] + 3, tb[3] + 2), fill=(0, 0, 0, 175))
        draw.text((box[0] + 6, box[1] + 5), text, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(base, layer).convert("RGB")


# -------------------------------------------------------------------- text / html
def alert_card_html(engine: DrishtiEngine, probs: np.ndarray) -> str:
    k = int(np.argmax(probs))
    cls, p = engine.classes[k], float(probs[k])
    name = engine.display.get(cls, cls)
    info = PLAYBOOK.get(cls, PLAYBOOK["normal"])
    if p < engine.threshold:
        status, color = "NEEDS HUMAN REVIEW", REVIEW_COLOR
        title = f"Uncertain - possibly {name.lower()}"
        action = (f"The calibrated confidence ({p:.0%}) is below the {engine.threshold:.0%} alert threshold. "
                  "Forward this frame to a human analyst before dispatching anyone.")
    else:
        status = "NO ALERT" if info["severity"] == "NONE" else f"{info['severity']} ALERT"
        color, title, action = info["color"], name, info["action"]
    return f"""
<div style="border-left:10px solid {color};background:#ffffff;color:#1f2937;padding:14px 18px;
            border-radius:10px;box-shadow:0 1px 6px rgba(0,0,0,.15);font-family:system-ui,sans-serif">
  <div style="font-size:12px;letter-spacing:.09em;font-weight:700;color:{color}">{html.escape(status)}</div>
  <div style="font-size:24px;font-weight:700;margin:2px 0 6px">{html.escape(title)}</div>
  <div style="font-size:15px;line-height:1.45">{html.escape(action)}</div>
  <div style="font-size:12px;color:#6b7280;margin-top:10px">Calibrated confidence {p:.1%} ·
       model {html.escape(engine.arch)} · alert threshold {engine.threshold:.0%}</div>
</div>"""


def scan_report_markdown(engine: DrishtiEngine, probs: np.ndarray) -> str:
    preds, conf = probs.argmax(1), probs.max(1)
    n = len(preds)
    confident = conf >= engine.threshold
    lines = ["| Class | Tiles | Share of area | Mean confidence | Severity |", "|---|---|---|---|---|"]
    order = sorted(range(len(engine.classes)), key=lambda i: PLAYBOOK.get(engine.classes[i], {}).get("rank", 9))
    for i in order:
        mask = preds == i
        if mask.any():
            cls = engine.classes[i]
            lines.append(f"| {engine.display.get(cls, cls)} | {int(mask.sum())} | {mask.mean():.0%} | "
                         f"{conf[mask].mean():.0%} | {PLAYBOOK.get(cls, {}).get('severity', '-')} |")
    hits = [engine.classes[i] for i in preds[confident] if engine.classes[i] != "normal"]
    if hits:
        worst = min(hits, key=lambda c: PLAYBOOK[c]["rank"])
        headline = (f"### ⚠️ {len(hits)} of {n} tiles ({len(hits) / n:.0%} of the area) show an emergency\n"
                    f"**Most severe:** {engine.display[worst]} - {PLAYBOOK[worst]['action']}")
    else:
        headline = f"### ✅ No confident emergency detected in {n} tiles"
    n_unsure = int((~confident).sum())
    note = (f"\n\n{n_unsure} tile(s) are below the {engine.threshold:.0%} confidence threshold (marked with '?') "
            "and should be checked by a person." if n_unsure else "")
    return headline + "\n\n" + "\n".join(lines) + note
