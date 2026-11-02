"""Streamlit app (v3): clinical-style chest X-ray pneumonia screening demo.

What's new compared with the first version
  * Hero header, card layout, risk gauge and plain-language verdict
  * Sensitivity presets (screening / balanced / strict) plus a custom cut-off
  * Image-quality checks (resolution, colour, brightness, contrast, shape)
  * Viewer tools: live heatmap opacity, contrast enhancement (CLAHE), invert
  * Batch mode with sortable results table and CSV export
  * Session history, downloadable PDF report, overlay PNG and JSON result
  * About tab with model card, how-to-read-the-heatmap and limitations

Run locally:   streamlit run app/app_v3.py
Optional:      put a few demo images in <project>/samples/ to get a sample picker.
"""
import html as htmllib
import io
import json
import re
import sys
import textwrap
from datetime import datetime
from hashlib import sha1
from pathlib import Path

# Make `src` importable no matter where Streamlit is launched from
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src import config
from src.predict import load_model_and_config, predict_image

st.set_page_config(
    page_title="Chest X-ray AI screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

DISPLAY_SIZE = 512          # images are upscaled for display only
SAMPLE_DIR = ROOT / "samples"
IMAGE_TYPES = ["jpg", "jpeg", "png"]
MAX_BATCH = 40
MIN_THR, MAX_THR = 0.01, 0.99
PRESETS = ["Balanced (model default)", "More sensitive", "More strict", "Custom"]

def _streamlit_version() -> tuple:
    try:
        return tuple(int(part) for part in st.__version__.split(".")[:2])
    except ValueError:
        return (0, 0)


# Newer Streamlit replaced use_container_width=True with width="stretch"; support both.
STRETCH = {"width": "stretch"} if _streamlit_version() >= (1, 50) else {"use_container_width": True}


DISCLAIMER = (
    "This project is for education and portfolio use only. It is not a medical device and "
    "must not be used to diagnose or treat any condition. Ask a qualified clinician to review any X-ray."
)

THEME_PRIMARY = "#0E7C86"
THEME_TOML = """\
# Premium light medical theme. Streamlit reads this file from the folder where you run `streamlit run`.
[theme]
base = "light"
primaryColor = "#0E7C86"
backgroundColor = "#F5F8FB"
secondaryBackgroundColor = "#E9F0F6"
textColor = "#0F2A43"
font = "sans serif"

[theme.sidebar]
backgroundColor = "#0B2A43"
secondaryBackgroundColor = "#14405F"
textColor = "#E8F0F7"
primaryColor = "#5CD6C8"

[client]
toolbarMode = "minimal"

[server]
maxUploadSize = 50
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&display=swap');
.block-container {padding-top: 2.4rem; max-width: 1250px;}
h1, h2, h3, .hero-title, .verdict-title {font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
  letter-spacing: -.01em;}
.hero {position: relative; overflow: hidden; color: #F4F8FB; border-radius: 22px; padding: 1.9rem 2.1rem 1.5rem;
  margin-bottom: 1.4rem; border: 1px solid rgba(255,255,255,.08); box-shadow: 0 18px 40px rgba(11,42,67,.22);
  background: radial-gradient(circle at 92% 8%, rgba(214,174,98,.24), transparent 38%),
              linear-gradient(135deg, #0B2A43 0%, #0F4660 58%, #0E7C86 118%);}
.eyebrow {font-size: .74rem; letter-spacing: .2em; text-transform: uppercase; color: #D6AE62; font-weight: 700;
  margin-bottom: .55rem;}
.hero-title {font-size: 2.35rem; font-weight: 700; line-height: 1.12; margin: 0; color: #FFFFFF;}
.hero-title span {background: linear-gradient(90deg, #F3DCA7, #D6AE62); -webkit-background-clip: text;
  background-clip: text; -webkit-text-fill-color: transparent;}
.hero-sub {margin: .65rem 0 1.05rem; color: rgba(244,248,251,.85); font-size: 1.02rem; max-width: 880px;}
.pill {display: inline-block; padding: .22rem .8rem; margin: 0 .45rem .3rem 0; border-radius: 999px; font-size: .8rem;
  font-weight: 600; color: #E8F0F7; border: 1px solid rgba(255,255,255,.26); background: rgba(255,255,255,.08);}
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: .8rem 0 1rem;}
.step {display: flex; gap: .8rem; align-items: flex-start; padding: 1rem 1.1rem; border-radius: 16px; color: #0F2A43;
  border: 1px solid #DCE6EF; background: #FFFFFF; box-shadow: 0 1px 2px rgba(15,42,67,.05), 0 10px 28px rgba(15,42,67,.06);}
.step-n {flex: 0 0 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-weight: 800; background: linear-gradient(135deg, #0B2A43, #0E7C86); color: #FFFFFF;}
.card {border: 1px solid #DCE6EF; border-radius: 16px; padding: 1rem 1.2rem; background: #FFFFFF; color: #0F2A43;
  margin-bottom: .8rem; box-shadow: 0 1px 2px rgba(15,42,67,.05), 0 10px 28px rgba(15,42,67,.06);}
.card-title {font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; color: #5B7186; font-weight: 600;
  margin-bottom: .4rem;}
.verdict {display: flex; gap: 1rem; align-items: flex-start; border-radius: 16px; padding: 1.1rem 1.3rem; color: #0F2A43;
  border: 1px solid; border-left-width: 8px; margin-bottom: .8rem;
  box-shadow: 0 1px 2px rgba(15,42,67,.05), 0 10px 28px rgba(15,42,67,.06);}
.v-pos {background: #FDF0F0; border-color: #D64545;}
.v-neg {background: #ECF8F2; border-color: #2E9E6A;}
.v-warn {background: #FFF7E6; border-color: #D9900A;}
.verdict-icon {font-size: 2rem; line-height: 1;}
.verdict-title {font-size: 1.4rem; font-weight: 700; margin-bottom: .25rem;}
.verdict-sub {color: #34506A; font-size: .97rem;}
.chip {display: inline-block; padding: .25rem .7rem; margin: 0 .4rem .4rem 0; border-radius: 999px; font-size: .84rem;
  color: #1E3A52; background: #F1F6FA; border: 1px solid #D5E1EB;}
.chip.ok {background: #E6F6EE; border-color: #BFE3CF;}
.chip.warn {background: #FFF3DB; border-color: #F0D699;}
.gauge-wrap {text-align: center;}
.gauge-wrap svg {width: 100%; max-width: 330px;}
.gauge-note {font-size: .82rem; color: #5B7186;}
[data-testid="stMetricValue"] {font-size: 1.5rem !important;}
.legend {display: flex; align-items: center; gap: .7rem; justify-content: center; font-size: .85rem; opacity: .85;
  margin-top: .3rem;}
.legend-bar {width: 220px; height: 12px; border-radius: 6px;
  background: linear-gradient(90deg,#00007f,#0000ff,#00ffff,#ffff00,#ff0000,#7f0000);}
.disclaimer {margin-top: 1.2rem; padding: .9rem 1.2rem; border-radius: 14px; border: 1px solid #E8C777;
  background: #FFF8E8; color: #5A3E0A; font-size: .92rem;}
@media (max-width: 800px) {.steps {grid-template-columns: 1fr;}}
</style>
"""

# Extra polish for native Streamlit widgets. Only applied when the light theme is active, because these rules
# assume a light page background.
EXTRA_CSS = """
<style>
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #DCE6EF; border-radius: 14px;
  padding: .7rem .9rem; box-shadow: 0 1px 2px rgba(15,42,67,.05);}
[data-testid="stFileUploaderDropzone"] {background: #FFFFFF; border: 1.5px dashed #9FB6C9; border-radius: 14px;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
.stButton > button, .stDownloadButton > button {border-radius: 10px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,.06);}
</style>
"""


# --------------------------------------------------------------------------- helpers
def render_html(markup: str) -> None:
    """Render raw HTML. Leading spaces and blank lines are stripped so Markdown never treats it as code."""
    markup = re.sub(r"^[ \t]+", "", markup, flags=re.M)
    markup = re.sub(r"\n\s*\n", "\n", markup)
    st.markdown(markup, unsafe_allow_html=True)


def esc(value) -> str:
    return htmllib.escape(str(value))


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model_and_config()


def load_image(data: bytes):
    """Return a fully loaded PIL image, or None when the bytes are not a readable image."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def to_uint8(array) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        if array.size and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    elif array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def upscale(image: np.ndarray, size: int = DISPLAY_SIZE) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def blend(base: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray:
    if heatmap.shape[:2] != base.shape[:2]:
        heatmap = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    return cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0)


def apply_view(image: np.ndarray, enhance: bool, invert: bool) -> np.ndarray:
    """Viewer-only adjustments. These never change the prediction."""
    out = image
    if enhance:
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    if invert:
        out = 255 - out
    return out


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


@st.cache_data(show_spinner=False, max_entries=128)
def run_inference(data: bytes, img_size: int, threshold: float, with_gradcam: bool) -> dict:
    model, _ = get_model()
    image = load_image(data)
    result = predict_image(
        model, image, img_size=img_size, threshold=threshold, with_gradcam=with_gradcam, alpha=0.4,
    )
    out = {
        "probability": float(result["probability"]),
        "flagged": result["label"] == config.POSITIVE_CLASS,
        "image": to_uint8(result["image"]),
    }
    if with_gradcam and result.get("heatmap_rgb") is not None:
        out["heatmap_rgb"] = to_uint8(result["heatmap_rgb"])
    return out


def margin_info(probability: float, threshold: float, flagged: bool):
    """How far the probability sits from the cut-off (0 = on the line, 1 = as far as it can be)."""
    if flagged:
        margin = (probability - threshold) / max(1e-6, 1.0 - threshold)
    else:
        margin = (threshold - probability) / max(1e-6, threshold)
    margin = float(min(max(margin, 0.0), 1.0))
    label = "Strong" if margin >= 0.5 else "Moderate" if margin >= 0.2 else "Borderline"
    return label, margin


def assess_quality(image: Image.Image, img_size: int):
    """Cheap sanity checks that catch the most common bad uploads. Returns [(level, message)]."""
    width, height = image.size
    thumb = image.convert("RGB")
    thumb.thumbnail((512, 512))
    rgb = np.asarray(thumb, dtype=np.float32)
    gray = rgb.mean(axis=-1)
    chroma = float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])))

    issues = []
    if min(width, height) < img_size:
        issues.append(("warn", f"Small image ({width} x {height}px): it is upscaled to {img_size} x {img_size}px, so fine detail is lost."))
    ratio = width / height
    if ratio < 0.6 or ratio > 1.7:
        issues.append(("warn", f"Unusual shape ({width} x {height}px): chest X-rays are usually close to square, so the image may be distorted when resized."))
    if chroma > 15:
        issues.append(("warn", "The image has strong colour. Chest X-rays are normally grayscale - is this a photo or a screenshot?"))
    if gray.mean() < 40:
        issues.append(("warn", "The image is very dark."))
    elif gray.mean() > 215:
        issues.append(("warn", "The image is very bright or over-exposed."))
    if gray.std() < 25:
        issues.append(("warn", "Low contrast: the lungs and ribs may not be distinguishable."))
    if not issues:
        issues.append(("ok", f"Image looks suitable ({width} x {height}px, grayscale, good contrast)."))
    return issues


def gauge_svg(probability: float, threshold: float, flagged: bool) -> str:
    """Semicircle gauge with a tick for the decision cut-off."""
    import math

    arc_length = math.pi * 80
    color = "#D64545" if flagged else "#2E9E6A"

    def point(radius, value):
        return 100 - radius * math.cos(math.pi * value), 100 - radius * math.sin(math.pi * value)

    x1, y1 = point(66, threshold)
    x2, y2 = point(94, threshold)
    return (
        '<svg viewBox="0 0 200 122" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#E3EAF1" stroke-width="16"/>'
        f'<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" '
        f'stroke-dasharray="{probability * arc_length:.2f} {arc_length + 40:.2f}"/>'
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="currentColor" stroke-width="3"/>'
        f'<text x="100" y="92" text-anchor="middle" font-size="27" font-weight="700" fill="currentColor">{probability:.1%}</text>'
        '<text x="100" y="110" text-anchor="middle" font-size="9" fill="currentColor" opacity=".7">pneumonia probability</text>'
        '<text x="20" y="118" text-anchor="middle" font-size="8" fill="currentColor" opacity=".6">0%</text>'
        '<text x="180" y="118" text-anchor="middle" font-size="8" fill="currentColor" opacity=".6">100%</text>'
        "</svg>"
    )


def verdict_copy(probability: float, threshold: float, flagged: bool, margin_label: str):
    """Return (css class, icon, title, subtitle)."""
    p, t = f"{probability:.1%}", f"{threshold:.2f}"
    borderline = margin_label == "Borderline"
    if flagged:
        title = "Possible pneumonia pattern (borderline)" if borderline else "Pneumonia pattern detected"
        sub = (f"The pneumonia probability ({p}) is above the cut-off of {t}. This is a screening signal, "
               "not a diagnosis - a clinician should review the X-ray.")
        css, icon = ("v-warn", "⚠️") if borderline else ("v-pos", "🔴")
    else:
        title = "No clear pneumonia pattern (borderline)" if borderline else "No pneumonia pattern detected"
        sub = (f"The pneumonia probability ({p}) is below the cut-off of {t}. A normal result does not rule "
               "out pneumonia - if there are symptoms, seek medical advice.")
        css, icon = ("v-warn", "⚠️") if borderline else ("v-neg", "🟢")
    if borderline:
        sub += " The result is close to the cut-off, so treat it as inconclusive."
    return css, icon, title, sub


def get_font(size: int, bold: bool = False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica-Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def build_report(*, case_id, notes, filename, probability, threshold, flagged, margin_label,
                 backbone, original, overlay, quality_msgs) -> bytes:
    """One-page A4 PDF summarising the analysis."""
    W, H, M = 1240, 1754, 70
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    ink, muted = (24, 24, 27), (110, 110, 120)
    tone = (185, 28, 28) if flagged else (21, 128, 61)
    fill = (254, 242, 242) if flagged else (240, 253, 244)

    d.rectangle([0, 0, W, 150], fill=(11, 42, 67))
    d.rectangle([0, 150, W, 156], fill=(214, 174, 98))
    d.text((M, 38), "Chest X-ray AI screening report", font=get_font(44, True), fill="white")
    d.text((M, 100), "Educational demo - not a medical device", font=get_font(24), fill=(214, 226, 238))

    y = 190
    meta = [
        ("Case ID", case_id or "-"),
        ("File", filename),
        ("Generated", datetime.now().strftime("%d %b %Y, %H:%M")),
        ("Model", f"{backbone} classifier with Grad-CAM"),
        ("Decision cut-off", f"{threshold:.2f}"),
    ]
    for key, value in meta:
        d.text((M, y), key, font=get_font(24), fill=muted)
        d.text((M + 270, y), str(value)[:62], font=get_font(24, True), fill=ink)
        y += 40

    y += 20
    d.rounded_rectangle([M, y, W - M, y + 170], radius=18, fill=fill, outline=tone, width=4)
    title = "Pneumonia pattern detected" if flagged else "No pneumonia pattern detected"
    d.text((M + 32, y + 26), title, font=get_font(40, True), fill=tone)
    d.text((M + 32, y + 94),
           f"Pneumonia probability {probability:.1%}   |   Cut-off {threshold:.2f}   |   Margin: {margin_label}",
           font=get_font(26), fill=ink)

    y += 210
    size = 540
    for x, img, caption in ((M, original, "Input X-ray"), (W - M - size, overlay, "Grad-CAM overlay")):
        tile = Image.fromarray(cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC))
        page.paste(tile, (x, y))
        d.text((x, y + size + 10), caption, font=get_font(22), fill=muted)

    y += size + 55
    d.text((M, y), "Image quality", font=get_font(28, True), fill=ink)
    y += 44
    for msg in quality_msgs[:3]:
        d.text((M, y), "- " + textwrap.shorten(msg, 78, placeholder="..."), font=get_font(22), fill=ink)
        y += 32

    y = 1385
    d.text((M, y), "Notes", font=get_font(28, True), fill=ink)
    y += 44
    note_lines = textwrap.wrap(" ".join((notes or "-").split()), 75)[:4]
    for line in note_lines:
        d.text((M, y), line, font=get_font(22), fill=ink)
        y += 32

    d.rounded_rectangle([M, H - 190, W - M, H - 60], radius=14, fill=(255, 251, 235), outline=(245, 158, 11), width=3)
    for i, line in enumerate(textwrap.wrap(DISCLAIMER, 78)[:3]):
        d.text((M + 24, H - 172 + i * 32), line, font=get_font(22), fill=(120, 53, 15))

    buffer = io.BytesIO()
    page.save(buffer, format="PDF", resolution=150.0)
    return buffer.getvalue()


def log_history(digest: str, name: str, probability: float, threshold: float, flagged: bool) -> None:
    seen = st.session_state.setdefault("seen", set())
    if digest in seen:
        return
    seen.add(digest)
    st.session_state.setdefault("history", []).append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "File": name,
        "Probability (%)": round(probability * 100, 1),
        "Cut-off": round(threshold, 2),
        "Result": "Pneumonia pattern" if flagged else "No pattern",
    })



def theme_helper() -> None:
    """Shown only when the premium theme from .streamlit/config.toml is not active."""
    with st.sidebar.expander("🎨 Enable the premium theme"):
        target = Path.cwd() / ".streamlit" / "config.toml"
        st.markdown(
            "The light medical colours come from a Streamlit config file, which is not active yet. "
            f"Streamlit reads it from the folder you run it in:\n\n`{target}`"
        )
        if target.exists():
            st.markdown("That file already exists, so copy these lines into it and restart Streamlit:")
            st.code(THEME_TOML, language="toml")
        elif st.button("Create the theme file for me"):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(THEME_TOML, encoding="utf-8")
                st.success("Created. Stop Streamlit (Ctrl+C) and run it again to apply the theme.")
            except OSError as exc:
                st.error(f"Could not write the file ({exc}). Create it by hand with the contents below.")
                st.code(THEME_TOML, language="toml")


# --------------------------------------------------------------------------- tabs
def scan_tab(cfg: dict, threshold: float) -> None:
    left, right = st.columns([1.35, 1], gap="large")
    data, name = None, None

    with left:
        uploaded = st.file_uploader("Upload a chest X-ray (frontal view, JPG or PNG)", type=IMAGE_TYPES, key="single")
        if uploaded is not None:
            data, name = uploaded.getvalue(), uploaded.name
        else:
            samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower().lstrip(".") in IMAGE_TYPES) \
                if SAMPLE_DIR.is_dir() else []
            if samples:
                choice = st.selectbox("...or try a bundled sample", ["None"] + [p.name for p in samples])
                if choice != "None":
                    data, name = (SAMPLE_DIR / choice).read_bytes(), choice

    with right:
        case_id = st.text_input("Case ID (optional)", placeholder="e.g. DEMO-001", max_chars=40)
        notes = st.text_area("Notes (optional)", height=90, max_chars=400,
                             placeholder="Shown in the PDF report only; they never affect the prediction.")
        st.caption("Demo only - please do not enter real patient identifiers.")

    if data is None:
        render_html("""
        <div class="steps">
        <div class="step"><div class="step-n">1</div><div><b>Upload</b><br>A frontal chest X-ray as JPG or PNG.</div></div>
        <div class="step"><div class="step-n">2</div><div><b>Analyse</b><br>The model estimates the probability of pneumonia.</div></div>
        <div class="step"><div class="step-n">3</div><div><b>Review</b><br>Check the heatmap, then download a PDF report.</div></div>
        </div>
        """)
        st.info("Upload an X-ray to get a prediction. Use the sidebar to change how sensitive the detection is.")
        return

    image = load_image(data)
    if image is None:
        st.error("This file could not be read as an image. Upload a JPG or PNG.")
        return

    with st.spinner("Analysing image..."):
        result = run_inference(data, cfg["img_size"], round(threshold, 2), True)

    probability, flagged = result["probability"], result["flagged"]
    margin_label, margin = margin_info(probability, threshold, flagged)
    quality = assess_quality(image, cfg["img_size"])
    digest = sha1(data).hexdigest()[:12]
    log_history(digest, name, probability, threshold, flagged)

    # ---- result summary
    st.divider()
    col_a, col_b = st.columns([1.25, 1], gap="large")
    with col_a:
        css, icon, title, sub = verdict_copy(probability, threshold, flagged, margin_label)
        render_html(f"""
        <div class="verdict {css}">
        <div class="verdict-icon">{icon}</div>
        <div><div class="verdict-title">{esc(title)}</div><div class="verdict-sub">{esc(sub)}</div></div>
        </div>
        """)
        chips = "".join(
            f'<span class="chip {level}">{"✅" if level == "ok" else "⚠️"} {esc(msg)}</span>' for level, msg in quality
        )
        render_html(f'<div class="card"><div class="card-title">Image quality check</div>{chips}</div>')
        if any(level == "warn" for level, _ in quality):
            st.caption("Quality warnings mean the result may be less reliable than usual.")

    with col_b:
        render_html(f"""
        <div class="card gauge-wrap">
        <div class="card-title">Pneumonia probability</div>
        {gauge_svg(probability, threshold, flagged)}
        <div class="gauge-note">The tick marks the decision cut-off ({threshold:.2f}).</div>
        </div>
        """)
        m1, m2, m3 = st.columns([1, 1, 1.4])
        m1.metric("Probability", f"{probability:.1%}")
        m2.metric("Cut-off", f"{threshold:.2f}")
        m3.metric("Margin", margin_label,
                  help="How far the probability is from the cut-off. This is not a calibrated confidence score.")

    # ---- viewer
    st.subheader("Image viewer")
    c1, c2, c3, c4 = st.columns([1, 1.5, 1, 1])
    has_heat = "heatmap_rgb" in result
    show_heatmap = c1.toggle("Grad-CAM heatmap", value=True, disabled=not has_heat)
    alpha = c2.slider("Heatmap opacity", 0.1, 0.9, 0.4, 0.05, disabled=not (has_heat and show_heatmap))
    enhance = c3.toggle("Enhance contrast", help="Viewer only (CLAHE). It does not change the prediction.")
    invert = c4.toggle("Invert", help="Viewer only. It does not change the prediction.")

    base = upscale(result["image"])
    view = apply_view(base, enhance, invert)
    heat = overlay = None
    if has_heat:
        heat = upscale(result["heatmap_rgb"])
        overlay = blend(view, heat, alpha)

    tab_side, tab_overlay, tab_orig = st.tabs(["Side by side", "Overlay", "Original upload"])
    with tab_side:
        if has_heat and show_heatmap:
            cols = st.columns(3)
            cols[0].image(view, caption="Input X-ray", **STRETCH)
            cols[1].image(heat, caption="Grad-CAM heatmap", **STRETCH)
            cols[2].image(overlay, caption="Overlay", **STRETCH)
        else:
            _, mid, _ = st.columns([1, 2, 1])
            mid.image(view, caption="Input X-ray", **STRETCH)
    with tab_overlay:
        if has_heat and show_heatmap:
            _, mid, _ = st.columns([1, 2, 1])
            mid.image(overlay, caption="Grad-CAM overlay", **STRETCH)
            render_html('<div class="legend"><span>Less influence</span><div class="legend-bar"></div>'
                        '<span>More influence</span></div>')
        else:
            st.info("Turn on the Grad-CAM heatmap above to see the overlay.")
    with tab_orig:
        _, mid, _ = st.columns([1, 2, 1])
        mid.image(image.convert("RGB"), caption=f"{name} ({image.size[0]} x {image.size[1]}px)",
                  **STRETCH)

    st.caption("Warm colours mark regions that pushed the model towards its prediction. "
               "A heatmap shows where the model looked, not a medical finding.")

    # ---- downloads
    st.subheader("Export")
    report_overlay = blend(base, upscale(result["heatmap_rgb"]), alpha) if has_heat else base
    pdf = build_report(
        case_id=case_id.strip(), notes=notes, filename=name, probability=probability, threshold=threshold,
        flagged=flagged, margin_label=margin_label, backbone=cfg.get("backbone", "unknown"),
        original=base, overlay=report_overlay, quality_msgs=[msg for _, msg in quality],
    )
    payload = {
        "file": name, "case_id": case_id.strip() or None, "pneumonia_probability": round(probability, 4),
        "cutoff": round(threshold, 2), "pneumonia_pattern_detected": bool(flagged), "margin": margin_label,
        "backbone": cfg.get("backbone", "unknown"), "generated_at": datetime.now().isoformat(timespec="seconds"),
        "disclaimer": DISCLAIMER,
    }
    d1, d2, d3 = st.columns(3)
    d1.download_button("📄 PDF report", pdf, file_name=f"xray_report_{digest}.pdf", mime="application/pdf",
                       **STRETCH, type="primary")
    d2.download_button("🖼️ Overlay image (PNG)", png_bytes(report_overlay), file_name=f"xray_overlay_{digest}.png",
                       mime="image/png", **STRETCH)
    d3.download_button("🧾 Result data (JSON)", json.dumps(payload, indent=2), file_name=f"xray_result_{digest}.json",
                       mime="application/json", **STRETCH)

    with st.expander("How to read this result"):
        st.markdown(
            "- **Probability** is the model's estimate that the image shows a pneumonia pattern. It is not a diagnosis.\n"
            "- **Cut-off** is the probability at or above which the image is flagged. Lower it to catch more cases "
            "(more false alarms); raise it to reduce false alarms (more missed cases).\n"
            "- **Margin** says how far the probability is from the cut-off. *Borderline* results are inconclusive.\n"
            "- **Heatmap** highlights the regions that most influenced the prediction. It can point at the wrong "
            "place even when the prediction is right, so never treat it as a finding."
        )


def batch_tab(cfg: dict, threshold: float) -> None:
    files = st.file_uploader(f"Upload several X-rays (up to {MAX_BATCH})", type=IMAGE_TYPES,
                             accept_multiple_files=True, key="batch")
    if not files:
        st.info("Add multiple images to screen them in one go. The cut-off from the sidebar is applied to all of them.")
        return
    if len(files) > MAX_BATCH:
        st.warning(f"Only the first {MAX_BATCH} images are analysed.")
        files = files[:MAX_BATCH]

    rows, previews = [], []
    bar = st.progress(0.0, text="Analysing...")
    for i, file in enumerate(files, start=1):
        data = file.getvalue()
        if load_image(data) is None:
            rows.append({"File": file.name, "Probability (%)": None, "Result": "Unreadable file", "Margin": "-"})
        else:
            res = run_inference(data, cfg["img_size"], round(threshold, 2), False)
            label, _ = margin_info(res["probability"], threshold, res["flagged"])
            rows.append({
                "File": file.name,
                "Probability (%)": round(res["probability"] * 100, 1),
                "Result": "Pneumonia pattern" if res["flagged"] else "No pattern",
                "Margin": label,
            })
            if len(previews) < 12:
                previews.append((file.name, res["probability"], res["flagged"], upscale(res["image"], 224)))
        bar.progress(i / len(files), text=f"Analysed {i} of {len(files)}")
    bar.empty()

    df = pd.DataFrame(rows)
    ok = df[df["Result"] != "Unreadable file"]
    flagged_n = int((ok["Result"] == "Pneumonia pattern").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images analysed", len(ok))
    m2.metric("Flagged", flagged_n)
    m3.metric("Not flagged", len(ok) - flagged_n)
    m4.metric("Mean probability", f"{ok['Probability (%)'].mean():.1f}%" if len(ok) else "-")

    order = st.radio("Sort by", ["Upload order", "Highest probability first"], horizontal=True)
    if order != "Upload order":
        df = df.sort_values("Probability (%)", ascending=False, na_position="last")
    st.dataframe(
        df, **STRETCH, hide_index=True,
        column_config={"Probability (%)": st.column_config.ProgressColumn(
            "Pneumonia probability", min_value=0, max_value=100, format="%.1f%%")},
    )
    st.download_button("⬇️ Download results (CSV)", df.to_csv(index=False), file_name="batch_results.csv",
                       mime="text/csv")

    if previews:
        with st.expander("Preview images"):
            cols = st.columns(4)
            for i, (fname, prob, flag, img) in enumerate(previews):
                cols[i % 4].image(img, caption=f"{'🔴' if flag else '🟢'} {fname} - {prob:.0%}",
                                  **STRETCH)


def history_tab() -> None:
    history = st.session_state.get("history", [])
    if not history:
        st.info("Analyses from this browser session will be listed here.")
        return
    df = pd.DataFrame(history)
    st.dataframe(
        df, **STRETCH, hide_index=True,
        column_config={"Probability (%)": st.column_config.ProgressColumn(
            "Pneumonia probability", min_value=0, max_value=100, format="%.1f%%")},
    )
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("⬇️ Download CSV", df.to_csv(index=False), file_name="session_history.csv", mime="text/csv")
    if c2.button("Clear history"):
        st.session_state["history"] = []
        st.session_state["seen"] = set()
        st.rerun()


def about_tab(cfg: dict) -> None:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("How it works")
        st.markdown(
            "1. **Pre-processing** - the X-ray is resized to the model's input size.\n"
            "2. **Classification** - a transfer-learning CNN estimates the probability of pneumonia.\n"
            "3. **Decision** - the image is flagged when the probability reaches the cut-off.\n"
            "4. **Explanation** - Grad-CAM shows which regions influenced the output."
        )
        st.subheader("Choosing a cut-off")
        st.markdown(
            "- **More sensitive** flags borderline images too - useful for screening, but gives more false alarms.\n"
            "- **More strict** flags only very confident cases - fewer false alarms, more missed cases.\n"
            "- **Balanced** uses the value tuned on validation data when the model was trained."
        )
    with right:
        st.subheader("Model card")
        card = {
            "Backbone": cfg.get("backbone", "unknown"),
            "Input size": f"{cfg['img_size']} x {cfg['img_size']} px",
            "Default cut-off": f"{float(cfg['threshold']):.3f}",
        }
        st.table(pd.DataFrame({"Value": card}))
        extras = {k: v for k, v in cfg.items()
                  if isinstance(v, (int, float, str)) and k not in {"backbone", "img_size", "threshold"}}
        if extras:
            with st.expander("All saved training settings"):
                st.json(extras)
        st.subheader("Limitations")
        st.markdown(
            "- Trained on one public dataset, so it may not generalise to other hospitals, scanners or age groups.\n"
            "- It only looks for a pneumonia pattern; it cannot tell bacterial from viral pneumonia or find other diseases.\n"
            "- Rotated, cropped, low-quality or non-X-ray images can give confident but meaningless results.\n"
            "- It has not been clinically validated and must never replace a clinician."
        )


# --------------------------------------------------------------------------- main
render_html(CSS)
theme_active = str(st.get_option("theme.primaryColor") or "").lower() == THEME_PRIMARY.lower()
if theme_active:
    render_html(EXTRA_CSS)

try:
    model, cfg = get_model()
except FileNotFoundError:
    st.error(
        f"No trained model found at `{config.MODEL_PATH.relative_to(ROOT)}`. "
        "Train one with `python -m src.train`, then reload this page."
    )
    st.stop()

base_threshold = float(min(max(cfg["threshold"], MIN_THR), MAX_THR))

with st.sidebar:
    st.markdown("### 🫁 Screening settings")
    preset = st.radio(
        "Detection sensitivity", PRESETS,
        help="An image is flagged as pneumonia when its probability is at or above the cut-off. "
             "A lower cut-off catches more cases but raises more false alarms.",
    )
    if preset == PRESETS[0]:
        threshold = base_threshold
    elif preset == PRESETS[1]:
        threshold = max(MIN_THR, base_threshold - 0.15)
    elif preset == PRESETS[2]:
        threshold = min(MAX_THR, base_threshold + 0.15)
    else:
        threshold = st.slider("Custom cut-off", MIN_THR, MAX_THR, value=base_threshold, step=0.01)
    st.caption(f"Cut-off in use: **{threshold:.2f}**")
    st.divider()
    st.markdown(
        f"**Backbone:** {esc(cfg.get('backbone', 'unknown'))}  \n"
        f"**Input size:** {cfg['img_size']} x {cfg['img_size']}  \n"
        f"**Analyses this session:** {len(st.session_state.get('history', []))}"
    )

render_html(f"""
<div class="hero">
<div class="eyebrow">AI-assisted radiology screening</div>
<div class="hero-title">Chest X-ray <span>AI screening</span></div>
<div class="hero-sub">Upload a chest X-ray. The model estimates the probability of pneumonia and highlights
the image regions that influenced its decision (Grad-CAM).</div>
<span class="pill">{esc(cfg.get('backbone', 'CNN'))}</span>
<span class="pill">Grad-CAM explainability</span>
<span class="pill">Demo - not a medical device</span>
</div>
""")

if not theme_active:
    theme_helper()

tab_scan, tab_batch, tab_history, tab_about = st.tabs(["🩻 Analyse", "🗂️ Batch", "🕘 History", "📚 About"])
with tab_scan:
    scan_tab(cfg, threshold)
with tab_batch:
    batch_tab(cfg, threshold)
with tab_history:
    history_tab()
with tab_about:
    about_tab(cfg)

render_html(f'<div class="disclaimer"><b>Important:</b> {esc(DISCLAIMER)}</div>')