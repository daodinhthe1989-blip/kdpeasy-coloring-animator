import io
import math
import random
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import cv2
import imageio
from PIL import Image

# ═══════════════════════════════════════════════════════════════════
# 🔐 SECURITY SETTINGS
# ═══════════════════════════════════════════════════════════════════
APP_PASSWORD = "KDPCOLOR2026"
BRAND_NAME = "KDPEasy Studio"
TOOL_NAME = "Coloring Animator"
WELCOME_MESSAGE = "Welcome, VIP Creator!"
# ═══════════════════════════════════════════════════════════════════

# Cap processing dimension — keeps region-detection and frame
# rendering fast and memory-safe on Streamlit Cloud's free tier.
MAX_PROCESS_DIM = 1100

st.set_page_config(
    page_title=f"{BRAND_NAME} — {TOOL_NAME}",
    page_icon="🎨",
    layout="wide",
)

CUSTOM_CSS = """
<style>
    .main > div { padding-top: 2rem; }
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%); }
    .block-container { max-width: 1200px; }
    h1 { color: #1f2937; font-weight: 700; }
    h2, h3 { color: #1f2937; }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover { background-color: #4338ca; color: white; }
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
    }
    .stDownloadButton>button:hover { background-color: #059669; color: white; }
    div[data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #cbd5e1;
    }
    .info-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #4f46e5;
        margin-bottom: 1rem;
    }
    .warn-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin-bottom: 1rem;
        color: #78350f;
    }
    .login-card {
        background: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        max-width: 480px;
        margin: 3rem auto;
        text-align: center;
    }
    .login-card h2 { color: #1f2937; margin-bottom: 0.5rem; }
    .login-card .brand {
        color: #4f46e5;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .preview-box {
        background: white;
        padding: 0.8rem;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 🔐 Password gate
# ═══════════════════════════════════════════════════════════════════
def check_password() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    st.markdown(
        f"""
        <div class="login-card">
            <div class="brand">{BRAND_NAME} · VIP Tool</div>
            <h2>🎨 {TOOL_NAME}</h2>
            <p style="color:#6b7280;margin-bottom:1.5rem;">
                Enter your VIP password to continue.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        pw = st.text_input("Password", type="password",
                           label_visibility="collapsed",
                           placeholder="Enter password…")
        ok = st.form_submit_button("🔓 Unlock", use_container_width=True)

    if ok:
        if pw == APP_PASSWORD:
            st.session_state.auth_ok = True
            st.rerun()
        else:
            st.error("❌ Wrong password. Please try again.")
    return False


if not check_password():
    st.stop()


# ═══════════════════════════════════════════════════════════════════
# 🧠 Core algorithm — region detection, coloring, rendering
# ═══════════════════════════════════════════════════════════════════
def load_and_align(line_bytes: bytes, colored_bytes: bytes,
                   max_dim: int = MAX_PROCESS_DIM
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Load both images, force them to the SAME size (the line-art's
    aspect ratio wins), and cap the long edge for fast processing.

    Width and height are rounded DOWN to the nearest even number —
    libx264 with yuv420p (used for the final MP4) requires both
    dimensions to be even, or ffmpeg exits immediately and every
    later write raises "Broken pipe".
    """
    line_img = Image.open(io.BytesIO(line_bytes)).convert("RGB")
    colored_img = Image.open(io.BytesIO(colored_bytes)).convert("RGB")

    w, h = line_img.size
    long_edge = max(w, h)
    if long_edge > max_dim:
        scale = max_dim / long_edge
        w, h = int(round(w * scale)), int(round(h * scale))
    w -= w % 2
    h -= h % 2

    line_img = line_img.resize((w, h), Image.LANCZOS)
    colored_img = colored_img.resize((w, h), Image.LANCZOS)
    return np.array(line_img), np.array(colored_img)


def detect_regions(line_rgb: np.ndarray, threshold: int = 128,
                   min_area_frac: float = 0.0015
                   ) -> Tuple[List[Dict], np.ndarray]:
    """Find enclosed fillable regions in the line-art using connected
    components — the same principle as a paint-bucket tool.

    A small dilation on the line mask first helps close hairline gaps
    common in AI-generated line art, so color doesn't leak between
    regions that were meant to be separate.

    Returns (kept_regions, all_fillable_mask). `all_fillable_mask`
    covers every non-line pixel, including tiny details filtered out
    of `kept_regions` for animation pacing (a highly detailed page can
    have hundreds of pixel-scale specks — animating each one looks
    chaotic). The caller uses `all_fillable_mask` to guarantee the
    finished frame still matches the reference exactly, via a final
    catch-all fill in render_frames.
    """
    h, w = line_rgb.shape[:2]
    gray = cv2.cvtColor(line_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # True non-line pixels, with NO dilation buffer — this is what the
    # catch-all fill uses, so it covers right up to the real line edge
    # instead of leaving a thin uncolored ring around every shape.
    all_fillable_mask = binary != 0

    # For REGION DETECTION only: thicken lines by 1px first to bridge
    # hairline gaps common in AI-generated line art before flood-filling.
    # This buffer is intentionally NOT used for all_fillable_mask above.
    line_mask = (binary == 0).astype(np.uint8) * 255
    line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)
    fillable = np.where(line_mask > 0, 0, 255).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(fillable, connectivity=4)

    min_area = max(20, int(h * w * min_area_frac))
    regions = []
    for label_id in range(1, num_labels):  # 0 = the line pixels themselves
        mask = labels == label_id
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(mask)
        cx, cy = int(xs.mean()), int(ys.mean())
        regions.append({
            "mask": mask,
            "area": area,
            "centroid": (cx, cy),
        })
    return regions, all_fillable_mask


def sample_colors(colored_rgb: np.ndarray, regions: List[Dict]) -> None:
    """Attach the median color of each region (sampled from the
    finished artwork) in place."""
    for r in regions:
        pixels = colored_rgb[r["mask"]]
        median = np.median(pixels, axis=0).astype(np.uint8)
        r["color"] = tuple(int(c) for c in median)


def order_regions(regions: List[Dict], mode: str) -> List[Dict]:
    ordered = list(regions)
    if mode == "largest":
        ordered.sort(key=lambda r: r["area"], reverse=True)
    elif mode == "top_down":
        ordered.sort(key=lambda r: (r["centroid"][1], r["centroid"][0]))
    else:  # "random"
        random.shuffle(ordered)
    return ordered


def render_and_encode_video(line_rgb: np.ndarray, colored_rgb: np.ndarray,
                            ordered_regions: List[Dict],
                            all_fillable_mask: np.ndarray,
                            duration_sec: float, fps: int, show_cursor: bool,
                            hold_sec: float = 1.5,
                            progress_cb=None) -> bytes:
    """Progressively reveal regions onto a persistent canvas, writing
    each frame straight to the video encoder as it's produced.

    Earlier versions built a Python list of every frame before
    encoding — for a 60s/20fps video that's 1200+ full-resolution
    frames held in memory at once, easily exceeding Streamlit Cloud's
    free-tier RAM and crashing the whole process. Streaming frames to
    the ffmpeg writer one at a time keeps memory flat regardless of
    video length.

    After the animated per-region reveal, a catch-all pass colors any
    remaining fine detail that was too small to animate individually
    (see detect_regions) — this guarantees the held final frames match
    the reference artwork exactly, even on very detailed pages.
    """
    h, w = line_rgb.shape[:2]
    n = len(ordered_regions)
    total_frames = max(1, int(duration_sec * fps))
    hold_frames = int(hold_sec * fps)
    grand_total = total_frames + hold_frames

    canvas = line_rgb.copy()
    revealed_so_far = 0
    cursor_radius = max(6, int(min(w, h) * 0.02))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    # imageio's 0-10 "quality" scale maps to an ESTIMATED bitrate, which
    # under-allocates bits for busy line-art detail (lace, foliage,
    # patterns) and shows up as smeared/ghosted edges. -crf directly
    # controls encoder quality regardless of content complexity — 18 is
    # visually near-lossless. -preset slow spends more effort for the
    # same size/quality tradeoff (fine for a one-off export, not a
    # live stream).
    writer = imageio.get_writer(
        tmp_path, fps=fps, codec="libx264", format="FFMPEG",
        macro_block_size=None,
        output_params=["-crf", "18", "-preset", "slow"],
    )
    try:
        for f in range(total_frames):
            progress = f / max(1, total_frames - 1)
            target_revealed = min(n, int(math.ceil(progress * n)))

            for i in range(revealed_so_far, target_revealed):
                mask = ordered_regions[i]["mask"]
                canvas[mask] = colored_rgb[mask]
            revealed_so_far = target_revealed

            frame = canvas.copy()
            if show_cursor and revealed_so_far > 0:
                cx, cy = ordered_regions[revealed_so_far - 1]["centroid"]
                cv2.circle(frame, (cx, cy), cursor_radius,
                          (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), max(2, cursor_radius // 3),
                          (60, 60, 60), thickness=-1, lineType=cv2.LINE_AA)
            writer.append_data(frame)
            if progress_cb and f % 5 == 0:
                progress_cb(f / grand_total)

        # Catch-all: fill any fine detail skipped by the per-region
        # animation so the finished artwork is a complete, exact match.
        canvas[all_fillable_mask] = colored_rgb[all_fillable_mask]

        # Hold the finished artwork for a beat before the video ends.
        for hf in range(hold_frames):
            writer.append_data(canvas)
            if progress_cb and hf % 5 == 0:
                progress_cb((total_frames + hf) / grand_total)
    finally:
        writer.close()

    with open(tmp_path, "rb") as f:
        data = f.read()
    return data


# ═══════════════════════════════════════════════════════════════════
# Header + logout
# ═══════════════════════════════════════════════════════════════════
hl, hr = st.columns([5, 1])
with hl:
    st.markdown(
        f"<h1>🎨 {TOOL_NAME}</h1>"
        f"<p style='color:#6b7280;margin-top:-0.5rem;'>"
        f"{BRAND_NAME} — turn a finished coloring page into a "
        f"satisfying 'coloring in progress' video.</p>",
        unsafe_allow_html=True,
    )
with hr:
    st.write("")
    if st.button("Logout", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
# Sidebar settings
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    duration_sec = st.slider("Video length (seconds)", 8, 60, 20, 1)

    order_label = st.selectbox(
        "Fill order",
        ["Largest areas first (recommended)", "Top to bottom", "Random"],
        index=0,
    )
    order_mode = {
        "Largest areas first (recommended)": "largest",
        "Top to bottom": "top_down",
        "Random": "random",
    }[order_label]

    show_cursor = st.checkbox("Show marker cursor", value=True,
                              help="Draws a small circle over the area "
                                   "being 'colored' right now.")

    with st.expander("🔧 Advanced"):
        line_threshold = st.slider(
            "Line detection threshold", 60, 200, 128, 5,
            help="Pixels darker than this are treated as ink lines. "
                 "Lower it if faint lines aren't being detected.",
        )
        min_area_pct = st.slider(
            "Ignore regions smaller than (% of image)", 0.05, 1.0, 0.15,
            0.05,
            help="Filters out tiny specks/anti-aliasing noise so the "
                 "video doesn't flicker.",
        )

    st.markdown("---")
    st.markdown(
        '<div class="info-card" style="font-size:0.85rem;">'
        "💡 <b>Tip:</b> Upload the exact same page in two versions — "
        "the black-and-white line art, and the fully colored artwork. "
        "Same composition, same crop."
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Upload
# ═══════════════════════════════════════════════════════════════════
st.markdown("### 1️⃣ Upload your two images")
col_up1, col_up2 = st.columns(2)
with col_up1:
    line_file = st.file_uploader(
        "Black & white line art (uncolored)",
        type=["png", "jpg", "jpeg", "webp"],
        key="line_art",
    )
with col_up2:
    colored_file = st.file_uploader(
        "Finished, fully colored version",
        type=["png", "jpg", "jpeg", "webp"],
        key="colored_art",
    )

if not line_file or not colored_file:
    st.info("👆 Upload both images to get started.")
    st.stop()

line_bytes = line_file.getvalue()
colored_bytes = colored_file.getvalue()

line_rgb, colored_rgb = load_and_align(line_bytes, colored_bytes)

st.markdown("### 2️⃣ Preview")
pcol1, pcol2 = st.columns(2)
with pcol1:
    st.markdown('<div class="preview-box">', unsafe_allow_html=True)
    st.image(line_rgb, caption="Line art", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with pcol2:
    st.markdown('<div class="preview-box">', unsafe_allow_html=True)
    st.image(colored_rgb, caption="Finished artwork", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Generate
# ═══════════════════════════════════════════════════════════════════
st.markdown("### 3️⃣ Generate the coloring animation")
go = st.button("🎬 Generate video", use_container_width=True)

if go:
    fps = 20
    progress = st.progress(0.0, text="Detecting fillable regions…")
    try:
        regions, all_fillable_mask = detect_regions(
            line_rgb, threshold=line_threshold,
            min_area_frac=min_area_pct / 100.0)
        if len(regions) < 2:
            st.markdown(
                '<div class="warn-card">⚠️ Only found '
                f'{len(regions)} fillable region(s). Your line art may '
                "have gaps that let colors leak together, or the "
                "threshold needs adjusting in Advanced settings.</div>",
                unsafe_allow_html=True,
            )
            st.stop()

        progress.progress(0.1, text=f"Found {len(regions)} regions — "
                                    "sampling colors…")
        sample_colors(colored_rgb, regions)
        ordered = order_regions(regions, order_mode)

        def _update(frac: float):
            progress.progress(0.15 + frac * 0.85,
                             text=f"Rendering & encoding video… "
                                  f"{int(frac * 100)}%")

        video_bytes = render_and_encode_video(
            line_rgb, colored_rgb, ordered, all_fillable_mask,
            duration_sec, fps, show_cursor, progress_cb=_update)

        progress.progress(1.0, text="Done!")
        st.success(f"✅ Video ready — {len(regions)} regions, "
                   f"{duration_sec:.0f}s, {len(video_bytes)/1024:.0f} KB")

        st.video(video_bytes)
        st.download_button(
            label="⬇️ Download coloring-animation.mp4",
            data=video_bytes,
            file_name="coloring-animation.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    except Exception as e:
        progress.empty()
        st.error(f"❌ Could not generate the video: {e}")


# ═══════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:#9ca3af;font-size:0.85rem;'>"
    f"{BRAND_NAME} — {TOOL_NAME} 🎨"
    f"</div>",
    unsafe_allow_html=True,
)
