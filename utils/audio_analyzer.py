import os
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
DURATION = 4


# ──────────────────────────────────────────────
# ARSITEKTUR — copy persis dari Colab
# ──────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class RawNetAudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1  = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1    = nn.BatchNorm1d(32)
        self.pool1  = nn.MaxPool1d(kernel_size=4)
        self.res1   = ResidualBlock1D(32)

        self.conv2  = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2    = nn.BatchNorm1d(64)
        self.pool2  = nn.MaxPool1d(kernel_size=4)
        self.res2   = ResidualBlock1D(64)

        self.conv3  = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3    = nn.BatchNorm1d(128)
        self.pool3  = nn.MaxPool1d(kernel_size=4)
        self.res3   = ResidualBlock1D(128)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1     = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.res2(x)
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.res3(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ──────────────────────────────────────────────
# PREPROCESSING — sama persis seperti training
# ──────────────────────────────────────────────

def extract_audio_from_video(video_path: str) -> str:
    import imageio_ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(SR), "-ac", "1",
        tmp.name
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        os.unlink(tmp.name)
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
    return tmp.name


def load_audio_waveform(audio_path: str) -> np.ndarray:
    import librosa
    waveform, _ = librosa.load(audio_path, sr=SR, mono=True)
    target = SR * DURATION
    if len(waveform) > target:
        waveform = waveform[:target]
    elif len(waveform) < target:
        waveform = np.pad(waveform, (0, target - len(waveform)), mode="constant")
    return waveform.astype(np.float32)


def preprocess_for_inference(audio_path: str) -> torch.Tensor:
    waveform = load_audio_waveform(audio_path)
    return torch.tensor(waveform).unsqueeze(0).unsqueeze(0)  # (1, 1, 64000)


# ──────────────────────────────────────────────
# ANALYZER
# ──────────────────────────────────────────────

class AudioDeepfakeAnalyzer:
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = self._load_model(model_path)

    def _load_model(self, model_path: str) -> RawNetAudioClassifier:
        model      = RawNetAudioClassifier(num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def analyze(self, video_path: str) -> dict:
        wav_path = None
        try:
            wav_path = extract_audio_from_video(video_path)

            if os.path.getsize(wav_path) < 1000:
                return self._no_audio_result()

            tensor = preprocess_for_inference(wav_path).to(self.device)

            with torch.no_grad():
                logits    = self.model(tensor)
                probs     = F.softmax(logits, dim=1)
                pred      = torch.argmax(probs, dim=1).item()
                prob_fake = probs[0, 1].item()
                prob_real = probs[0, 0].item()

            issues = []
            if pred == 1:
                if prob_fake > 0.85:
                    issues.append("Audio dengan keyakinan tinggi terdeteksi sebagai AI-generated / deepfake")
                else:
                    issues.append("Audio kemungkinan AI-generated atau telah dimanipulasi")
            elif prob_real < 0.65:
                issues.append("Audio asli namun confidence rendah – perlu pemeriksaan lebih lanjut")

            return {
                "score":     round(prob_real * 100, 2),
                "status":    "fail" if prob_fake > 0.65 else ("warning" if prob_fake > 0.4 else "pass"),
                "is_fake":   pred == 1,
                "prob_real": round(prob_real, 4),
                "prob_fake": round(prob_fake, 4),
                "issues":    issues,
                "model":     "RawNetAudioClassifier",
            }

        except RuntimeError as e:
            if "ffmpeg" in str(e).lower() or "audio" in str(e).lower():
                return self._no_audio_result()
            raise
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

    @staticmethod
    def _no_audio_result() -> dict:
        return {
            "score":     50,
            "status":    "warning",
            "is_fake":   None,
            "prob_real": None,
            "prob_fake": None,
            "issues":    ["Tidak ada audio track yang dapat dianalisis dalam video ini"],
            "model":     "RawNetAudioClassifier",
        }