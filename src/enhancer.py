"""
Frequency-Aware Adaptive Filter Fusion with Iterative Quality Optimization
CS-712 Image Processing - Term Project
"""

import cv2
import numpy as np
from scipy.optimize import minimize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
warnings.filterwarnings('ignore')


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img.astype(np.float32) / 255.0


# Three filters in our fusion bank
def apply_gaussian(patch):
    p = (patch * 255).astype(np.uint8)
    return cv2.GaussianBlur(p, (5, 5), 1.0).astype(np.float32) / 255.0

def apply_median(patch):
    p = (patch * 255).astype(np.uint8)
    return cv2.medianBlur(p, 3).astype(np.float32) / 255.0

def apply_unsharp(patch, strength=1.5):
    p = (patch * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(p, (5, 5), 1.0)
    sharpened = cv2.addWeighted(p, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.float32) / 255.0


# ── BASELINES (all from course lectures) ─────────────────────────────

def baseline_gaussian(img):
    """Gaussian Low-Pass Filter — Lecture 13"""
    p = (img * 255).astype(np.uint8)
    return cv2.GaussianBlur(p, (5, 5), 1.0).astype(np.float32) / 255.0

def baseline_unsharp_mask(img):
    """Unsharp Masking — Lecture 15"""
    p = (img * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(p, (5, 5), 1.0)
    sharp = cv2.addWeighted(p, 2.5, blurred, -1.5, 0)
    return np.clip(sharp, 0, 255).astype(np.float32) / 255.0

def baseline_laplacian(img):
    """Laplacian Sharpening — Lecture 19"""
    p = (img * 255).astype(np.uint8)
    lap = cv2.Laplacian(p, cv2.CV_64F)
    sharpened = p.astype(np.float64) - lap
    return np.clip(sharpened, 0, 255).astype(np.float32) / 255.0

def baseline_gaussian_highpass(img):
    """Gaussian High-Pass Filter — Lecture 14 (H_HP = 1 - H_LP)"""
    p = img.astype(np.float32)
    h, w = p.shape
    u = np.fft.fftfreq(h) * h
    v = np.fft.fftfreq(w) * w
    V, U = np.meshgrid(v, u)
    D2 = U**2 + V**2
    H_LP = np.exp(-D2 / (2 * 30.0**2))
    H_HP = 1 - H_LP
    F = np.fft.fft2(p)
    enhanced = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(F) * (1 + 0.5 * H_HP)))
    return np.clip(np.abs(enhanced), 0, 1).astype(np.float32)


# ── Core algorithm ────────────────────────────────────────────────────

def analyze_patch_frequency(patch):
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = patch.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4
    y, x = np.ogrid[:h, :w]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    low_energy = np.sum(magnitude[mask])
    total_energy = np.sum(magnitude) + 1e-10
    return float(np.clip(1.0 - (low_energy / total_energy), 0, 1))


def get_starting_weights(patch):
    variance = float(np.var(patch))
    high_freq_ratio = analyze_patch_frequency(patch)
    edge_strength = float(np.mean(np.abs(cv2.Laplacian(
        (patch * 255).astype(np.uint8), cv2.CV_64F))))
    edge_norm = min(edge_strength / 50.0, 1.0)
    w_median   = 0.2 + 0.4 * (1 - high_freq_ratio) * variance * 10
    w_gaussian = 0.2 + 0.4 * (1 - edge_norm)
    w_sharp    = 0.1 + 0.6 * high_freq_ratio * edge_norm
    total = w_median + w_gaussian + w_sharp + 1e-10
    return np.array([w_median, w_gaussian, w_sharp]) / total


def fuse_filters(patch, w):
    fused = w[0]*apply_median(patch) + w[1]*apply_gaussian(patch) + w[2]*apply_unsharp(patch)
    return np.clip(fused, 0, 1).astype(np.float32)


def optimize_weights(patch, ref_patch, starting_weights):
    best = {'score': -1, 'weights': starting_weights.copy()}

    def objective(w_raw):
        w = np.exp(w_raw) / (np.sum(np.exp(w_raw)) + 1e-10)
        fused = fuse_filters(patch, w)
        win = min(fused.shape[0], fused.shape[1], 7)
        if win % 2 == 0: win -= 1
        if win < 3:
            score = -float(np.mean((fused - ref_patch)**2))
        else:
            score = ssim(ref_patch, fused, data_range=1.0, win_size=win)
        if score > best['score']:
            best['score'] = score
            best['weights'] = w.copy()
        return -score

    minimize(objective, np.log(starting_weights + 1e-10), method='Nelder-Mead',
             options={'maxiter': 80, 'xatol': 0.01, 'fatol': 0.01})
    return best['weights']


def enhance_image(noisy_img, ref_img=None, patch_size=16, progress_cb=None):
    H, W = noisy_img.shape
    ref = ref_img if ref_img is not None else noisy_img
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    noisy_pad = np.pad(noisy_img, ((0, pad_h), (0, pad_w)), mode='reflect')
    ref_pad   = np.pad(ref,       ((0, pad_h), (0, pad_w)), mode='reflect')
    out_pad   = np.zeros_like(noisy_pad)
    rows = noisy_pad.shape[0] // patch_size
    cols = noisy_pad.shape[1] // patch_size
    done = 0
    total = rows * cols

    # Track weights and improvement per patch
    all_weights = []
    improvement_map = np.zeros((rows, cols), dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            r0, r1 = i*patch_size, (i+1)*patch_size
            c0, c1 = j*patch_size, (j+1)*patch_size
            patch = noisy_pad[r0:r1, c0:c1]
            ref_p = ref_pad[r0:r1, c0:c1]
            w0 = get_starting_weights(patch)
            w_opt = optimize_weights(patch, ref_p, w0)
            enhanced_patch = fuse_filters(patch, w_opt)
            out_pad[r0:r1, c0:c1] = enhanced_patch

            # Record weights
            all_weights.append(w_opt)

            # Improvement = SSIM(enhanced) - SSIM(noisy)
            # Both compared against a smoothed reference (local mean)
            win = min(patch_size, 7)
            if win % 2 == 0: win -= 1
            if win >= 3:
                smooth_ref = cv2.GaussianBlur(patch, (win, win), 0)
                ssim_before = ssim(smooth_ref, patch,          data_range=1.0, win_size=win)
                ssim_after  = ssim(smooth_ref, enhanced_patch, data_range=1.0, win_size=win)
                improvement_map[i, j] = ssim_after - ssim_before
            else:
                improvement_map[i, j] = 0.0

            done += 1
            if progress_cb: progress_cb(done / total)

    enhanced = np.clip(out_pad[:H, :W], 0, 1).astype(np.float32)
    avg_weights = np.mean(all_weights, axis=0)
    return enhanced, avg_weights, improvement_map


def evaluate(original, enhanced):
    win = min(original.shape[0], original.shape[1], 7)
    if win % 2 == 0: win -= 1
    if win < 3:
        return {'PSNR': 0, 'SSIM': 0, 'MSE': round(float(np.mean((original-enhanced)**2)), 6)}
    return {
        'PSNR': round(float(psnr(original, enhanced, data_range=1.0)), 2),
        'SSIM': round(float(ssim(original, enhanced, data_range=1.0, win_size=win)), 4),
        'MSE':  round(float(np.mean((original-enhanced)**2)), 6)
    }
