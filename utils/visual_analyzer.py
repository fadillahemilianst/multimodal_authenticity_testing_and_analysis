import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

NUM_FRAMES   = 10
IMAGE_SIZE   = 224
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ──────────────────────────────────────────────
# TRANSFORM — sama dengan eval_transform di Colab
# ──────────────────────────────────────────────

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ──────────────────────────────────────────────
# PREPROCESSING — copy persis dari Colab
# ──────────────────────────────────────────────

def crop_face_or_center(image):
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        x, y, w, h = faces[0]
        margin = int(0.2 * max(w, h))
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, image.shape[1])
        y2 = min(y + h + margin, image.shape[0])
        return image[y1:y2, x1:x2]

    # fallback: center crop
    h, w, _ = image.shape
    min_side = min(h, w)
    start_x  = (w - min_side) // 2
    start_y  = (h - min_side) // 2
    return image[start_y:start_y + min_side, start_x:start_x + min_side]


def extract_frames(video_path: str, num_frames: int = NUM_FRAMES):
    """
    Extract num_frames frame tersebar merata dari video.
    Return list of PIL Images (sudah di-crop face / center).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Video tidak memiliki frame yang dapat dibaca")

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        cropped = crop_face_or_center(frame)
        resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))

        # OpenCV BGR → PIL RGB
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        frames.append(pil_img)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Tidak ada frame yang berhasil diekstrak")

    return frames


# ──────────────────────────────────────────────
# MODEL — EfficientNet-B0, sama seperti Colab
# ──────────────────────────────────────────────

def build_model(num_classes: int = 2) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, num_classes)
    )
    return model


# ──────────────────────────────────────────────
# ANALYZER
# ──────────────────────────────────────────────

class VisualDeepfakeAnalyzer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model(model_path)

    def _load_model(self, model_path: str) -> nn.Module:
        model      = build_model(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def analyze(self, video_path: str) -> dict:
        try:
            frames = extract_frames(video_path, num_frames=NUM_FRAMES)
        except RuntimeError as e:
            return {
                "score":          50,
                "status":         "warning",
                "is_fake":        None,
                "prob_real_mean": None,
                "prob_fake_mean": None,
                "frames_analyzed": 0,
                "votes_fake":     0,
                "votes_real":     0,
                "issues":         [f"Analisis visual gagal: {str(e)}"],
                "model":          "EfficientNet-B0",
            }

        all_prob_real = []
        all_prob_fake = []
        votes_fake    = 0
        votes_real    = 0

        with torch.no_grad():
            for pil_img in frames:
                tensor = eval_transform(pil_img).unsqueeze(0).to(self.device)  # (1, 3, 224, 224)
                logits = self.model(tensor)
                probs  = F.softmax(logits, dim=1)

                prob_real = probs[0, 0].item()
                prob_fake = probs[0, 1].item()

                all_prob_real.append(prob_real)
                all_prob_fake.append(prob_fake)

                if prob_fake > prob_real:
                    votes_fake += 1
                else:
                    votes_real += 1

        mean_prob_real = float(np.mean(all_prob_real))
        mean_prob_fake = float(np.mean(all_prob_fake))

        # Majority voting
        is_fake = votes_fake > votes_real

        issues = []
        if is_fake:
            if votes_fake >= NUM_FRAMES * 0.8:
                issues.append("Visual dengan keyakinan tinggi terdeteksi sebagai deepfake")
            else:
                issues.append("Visual kemungkinan dimanipulasi – mayoritas frame terdeteksi fake")
        elif mean_prob_real < 0.65:
            issues.append("Visual asli namun confidence rendah – perlu pemeriksaan lebih lanjut")

        return {
            "score":           round(mean_prob_real * 100, 2),
            "status":          "fail" if (is_fake and mean_prob_fake > 0.65) else
                               ("warning" if mean_prob_fake > 0.4 else "pass"),
            "is_fake":         is_fake,
            "prob_real":  round(mean_prob_real, 4),
            "prob_fake":  round(mean_prob_fake, 4),
            "frames_analyzed": len(frames),
            "votes_fake":      votes_fake,
            "votes_real":      votes_real,
            "issues":          issues,
            "model":           "EfficientNet-B0",
        }