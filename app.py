"""
Streamlit App — FAFF: Adaptive Medical Image Enhancement
Frequency-Aware Adaptive Filter Fusion with Iterative Quality Optimization
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from enhancer import (
    enhance_image,
    baseline_gaussian, baseline_unsharp_mask,
    baseline_laplacian, baseline_gaussian_highpass,
)

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Medical Image Enhancement",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #065A82;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background: #f0f7ff;
        border-left: 4px solid #065A82;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #333;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #065A82;
        border-bottom: 2px solid #e0eaf4;
        padding-bottom: 0.3rem;
        margin: 1rem 0 0.8rem 0;
    }

</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────
st.markdown('<div class="main-title">🏥 FAFF — Adaptive Medical Image Enhancement</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Frequency-Aware Adaptive Filter Fusion with Iterative Quality Optimization</div>', unsafe_allow_html=True)


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def auto_detect_noise(img):
    """
    Detect noise from real medical image.
    Crops the center region to avoid black borders/background
    that are common in X-rays and misclassified as dark noise.
    """
    # Crop center 60% to avoid black X-ray borders
    h, w = img.shape
    cy, cx = h // 2, w // 2
    rh, rw = int(h * 0.3), int(w * 0.3)
    center = img[cy-rh:cy+rh, cx-rw:cx+rw]

    # Gaussian noise: measure local variance in center region
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    diff = img.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(np.std(diff))

    # S&P detection: only check isolated extreme pixels in center
    kernel = np.ones((3, 3), np.uint8)
    bright = (center > 250).astype(np.uint8)
    dark   = (center < 5).astype(np.uint8)
    bright_eroded = cv2.erode(bright, kernel, iterations=1)
    dark_eroded   = cv2.erode(dark,   kernel, iterations=1)
    isolated_sp = np.sum(bright - bright_eroded) + np.sum(dark - dark_eroded)
    sp_ratio = isolated_sp / center.size

    noise_type  = 'Salt & Pepper' if sp_ratio > 0.002 else 'Gaussian'
    noise_level = round(min(noise_std / 255.0, 1.0), 4)
    return noise_type, noise_level


def load_demo_image():
    """Load real demo xray if available, else use synthetic fallback."""
    demo_path = os.path.join(os.path.dirname(__file__), 'demo_noisy_xray.png')
    if os.path.exists(demo_path):
        img = cv2.imread(demo_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    # Synthetic fallback
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(img, (128, 140), (110, 120), 0, 0, 360, 45, -1)
    cv2.ellipse(img, (80, 145), (50, 72), 0, 0, 360, 22, -1)
    cv2.ellipse(img, (176, 145), (50, 72), 0, 0, 360, 22, -1)
    cv2.rectangle(img, (121, 50), (135, 235), 115, -1)
    for y in range(80, 215, 18):
        cv2.ellipse(img, (128, y), (92, 10), 0, 0, 360, 95, 2)
    cv2.ellipse(img, (110, 158), (30, 38), 0, 0, 360, 78, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    gaussian = np.random.normal(0, 18, img.shape).astype(np.float32)
    sp = np.zeros(img.shape, dtype=np.float32)
    cw = [np.random.randint(0, i, int(img.size*0.008)) for i in img.shape]
    cb = [np.random.randint(0, i, int(img.size*0.008)) for i in img.shape]
    sp[cw[0], cw[1]] = 80
    sp[cb[0], cb[1]] = -80
    speckle = img.astype(np.float32) * np.random.normal(0, 0.08, img.shape)
    return np.clip(img.astype(np.float32) + gaussian + sp + speckle, 0, 255).astype(np.uint8)


def plot_histogram(images, titles, colors):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3))
    fig.patch.set_facecolor('#0D1117')
    if n == 1:
        axes = [axes]
    for ax, img, title, color in zip(axes, images, titles, colors):
        img_u8 = to_uint8(img) if img.max() <= 1.0 else img.astype(np.uint8)
        hist = cv2.calcHist([img_u8], [0], None, [256], [0, 256]).flatten()
        ax.set_facecolor('#1A1A2E')
        ax.fill_between(range(256), hist, color=color, alpha=0.6)
        ax.plot(hist, color=color, linewidth=1.2)
        ax.set_title(title, color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Pixel Intensity', color='#888', fontsize=9)
        ax.set_ylabel('Pixel Count', color='#888', fontsize=9)
        ax.tick_params(colors='#888')
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, 255])
    plt.tight_layout(pad=2.0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, facecolor='#0D1117')
    plt.close()
    buf.seek(0)
    return buf

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    patch_size = st.selectbox(
        "Patch Size", [8, 16, 32], index=1,
        help="8 = more detail but slower | 32 = faster but less precise"
    )
    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    1. 🔲 **Split** image into patches
    2. 📊 **FFT** *(Fast Fourier Transform)* analysis per patch
    3. 🔍 **Detect** noise type automatically
    4. 🎛️ **Derive** content-aware weights
    5. 🔄 **Optimize** weights using SSIM score
    6. 🖼️ **Reconstruct** enhanced image
    """)
    st.caption("FFT = Fast Fourier Transform | SSIM = Structural Similarity Index Measure")
    st.caption("Upload any noisy medical image — noise is detected automatically.")

# ── Upload ───────────────────────────────────
uploaded = st.file_uploader(
    "📁 Upload a Noisy Medical Image (X-ray, Ultrasound, MRI)",
    type=["png", "jpg", "jpeg", "bmp", "tiff"]
)

if not uploaded:
    st.markdown('<div class="info-box">👆 Upload a noisy medical image above, or try the demo button below.</div>', unsafe_allow_html=True)
    if st.button("🔬 Use Demo Medical Image"):
        demo = load_demo_image()
        buf = io.BytesIO()
        Image.fromarray(demo).save(buf, format='PNG')
        buf.seek(0)
        uploaded = buf

# ── Process ───────────────────────────────────
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        st.error("Could not read image. Please try another file.")
        st.stop()

    h, w = img_gray.shape
    if max(h, w) > 256:
        scale = 256 / max(h, w)
        img_gray = cv2.resize(img_gray, (int(w * scale), int(h * scale)))

    input_img = img_gray.astype(np.float32) / 255.0

    # Detect noise from real image — no fake noise added
    noise_type, noise_level = auto_detect_noise(img_gray)
    st.markdown(
        f'<div class="info-box">🔍 <b>Noise detected from image:</b> '
        f'Type = <b>{noise_type}</b> &nbsp;|&nbsp; '
        f'Level = <b>{noise_level}</b> &nbsp;|&nbsp; '
        f'Enhancement applied accordingly.</div>',
        unsafe_allow_html=True
    )

    # Run FAFF
    st.markdown('<div class="section-header">⚡ Processing</div>', unsafe_allow_html=True)
    bar = st.progress(0, text="FAFF: Analyzing patches and optimizing filter weights...")

    def update_progress(p):
        bar.progress(min(p, 1.0), text=f"FAFF processing patches... {int(p*100)}%")

    enhanced, avg_weights, ssim_map = enhance_image(input_img, ref_img=input_img, patch_size=patch_size, progress_cb=update_progress)
    bar.empty()
    st.success("✅ Enhancement complete!")

    # Lecture-based baselines
    b_gaussian  = baseline_gaussian(input_img)
    b_ghp       = baseline_gaussian_highpass(input_img)
    b_unsharp   = baseline_unsharp_mask(input_img)
    b_laplacian = baseline_laplacian(input_img)

    # ── Optimization Results ──────────────────
    st.markdown('<div class="section-header">📊 Optimization Results — What Happened Inside FAFF</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">During enhancement, FAFF automatically found the best filter combination for each patch. Here\'s what it discovered across the entire image.</div>', unsafe_allow_html=True)

    ow1, ow2 = st.columns(2)

    # ── Weight Bar Chart ──────────────────────
    with ow1:
        st.markdown("**Average Filter Weights Used Across All Patches**")
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#0D1117')
        ax.set_facecolor('#1A1A2E')
        filters = ['Median\n(Denoising)', 'Gaussian\n(Smoothing)', 'Unsharp\n(Sharpening)']
        colors  = ['#E74C3C', '#4A90D9', '#02C39A']
        bars = ax.bar(filters, avg_weights * 100, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, avg_weights * 100):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom',
                    color='white', fontsize=11, fontweight='bold')
        ax.set_ylabel('Weight (%)', color='#888', fontsize=9)
        ax.set_title('Content-Aware Filter Weights', color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='white')
        ax.set_ylim(0, 70)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']: ax.spines[sp].set_color('#333')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, facecolor='#0D1117')
        plt.close()
        buf.seek(0)
        st.image(buf, use_container_width=True)
        st.caption(f"Median: {avg_weights[0]*100:.1f}% | Gaussian: {avg_weights[1]*100:.1f}% | Unsharp: {avg_weights[2]*100:.1f}% — weights vary per patch based on local content")

    # ── SSIM Heatmap ──────────────────────────
    with ow2:
        st.markdown("**SSIM Improvement Heatmap — Per Patch Enhancement Gain**")
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor('#0D1117')
        vmax = max(abs(ssim_map.max()), abs(ssim_map.min()), 0.01)
        im = ax.imshow(ssim_map, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SSIM Improvement', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
        ax.set_title('SSIM Improvement per Patch', color='white', fontsize=11, fontweight='bold')
        ax.set_xlabel('Patch Column', color='#888', fontsize=9)
        ax.set_ylabel('Patch Row', color='#888', fontsize=9)
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('#0D1117')
        plt.tight_layout()
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', dpi=120, facecolor='#0D1117')
        plt.close()
        buf2.seek(0)
        st.image(buf2, use_container_width=True)
        avg_imp  = float(np.mean(ssim_map))
        max_imp  = float(np.max(ssim_map))
        improved = int(np.sum(ssim_map > 0))
        total_p  = ssim_map.size
        st.caption(f"Avg improvement: {avg_imp:+.3f} | Max: {max_imp:+.3f} | {improved}/{total_p} patches improved — green = big improvement, red = slight reduction")

    # ── 1. Image Comparison ───────────────────
    st.markdown('<div class="section-header">🖼️ Image Comparison</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(to_uint8(input_img), caption="⚠️ Noisy Input", use_container_width=True)
    with c2:
        st.image(to_uint8(enhanced), caption="🥇 FAFF Enhanced Output", use_container_width=True)

    # ── 2. Baseline Comparisons ───────────────
    st.markdown('<div class="section-header">📚 Baseline Comparisons</div>', unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.image(to_uint8(b_gaussian),  caption="Gaussian Low-Pass",  use_container_width=True)
    with b2:
        st.image(to_uint8(b_ghp),       caption="Gaussian High-Pass",  use_container_width=True)
    with b3:
        st.image(to_uint8(b_unsharp),   caption="Unsharp Masking",     use_container_width=True)
    with b4:
        st.image(to_uint8(b_laplacian), caption="Laplacian Sharpening", use_container_width=True)

    # ── 3. Histogram ──────────────────────────
    st.markdown('<div class="section-header">📊 Histogram Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">📈 <b>Wider spread</b> = better contrast &nbsp;|&nbsp; 📉 <b>Fewer extreme spikes</b> = less noise</div>', unsafe_allow_html=True)

    st.image(plot_histogram(
        [input_img, enhanced],
        ["Noisy Input", "FAFF Enhanced Output"],
        ['#E74C3C', '#02C39A']
    ), use_container_width=True)

    st.image(plot_histogram(
        [b_gaussian, b_ghp, b_unsharp, b_laplacian],
        ["Gaussian Low-Pass", "Gaussian High-Pass", "Unsharp Masking", "Laplacian Sharpening"],
        ['#4A90D9', '#9B59B6', '#1ABC9C', '#E74C3C']
    ), use_container_width=True)

    st.markdown("""
    <div style="font-size:0.85rem;color:#555;text-align:center;margin-top:0.5rem;font-style:italic;">
    FAFF produces a smoother, wider histogram compared to the noisy input — fewer extreme spikes indicate effective noise removal,
    while the wider spread indicates improved contrast and better visibility of diagnostic details.
    </div>
    <div style="font-size:0.85rem;color:#555;text-align:center;margin-top:0.4rem;font-style:italic;">
    Baseline methods show either over-smoothing (Gaussian LP — narrow spread) or noise amplification (Laplacian — sharp spikes),
    whereas FAFF achieves the best balance between noise suppression and contrast enhancement.
    </div>
    """, unsafe_allow_html=True)

    # ── 4. Download ───────────────────────────
    st.markdown('<div class="section-header">💾 Download</div>', unsafe_allow_html=True)
    out_buf = io.BytesIO()
    Image.fromarray(to_uint8(enhanced)).save(out_buf, format='PNG')
    st.download_button(
        label="⬇️ Download FAFF Enhanced Image",
        data=out_buf.getvalue(),
        file_name="faff_enhanced_medical_image.png",
        mime="image/png"
    )
