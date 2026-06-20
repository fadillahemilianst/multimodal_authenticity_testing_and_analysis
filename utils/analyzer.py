import cv2
import numpy as np
from collections import defaultdict
import os

AUDIO_MODEL_PATH = "model/best_rawnet_audio_binary.pth"
VISUAL_MODEL_PATH = "model/best_efficientnet_visual_binary_full_extracted3.pth"  

class VideoAuthenticityAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.cap = None
        self.results = {}

    def analyze(self):
        self.cap = cv2.VideoCapture(self.filepath)

        if not self.cap.isOpened():
            raise ValueError("Cannot open video file")

        try:
            metadata = self._extract_metadata()
            frames = self._extract_sample_frames()

            if len(frames) == 0:
                raise ValueError("Could not extract frames from video")

            checks = {
                'audio': self._analyze_audio(), 
                'visual': self._analyze_visual(),
            }

            score, verdict, confidence = self._calculate_verdict(checks)

            return {
                'verdict': verdict,
                'authenticity_score': score,
                'confidence': confidence,
                'metadata': metadata,
                'checks': checks,
                'summary': self._generate_summary(checks)
            }
        finally:
            self.cap.release()

    def _analyze_audio(self) -> dict:
        if not os.path.exists(AUDIO_MODEL_PATH):
            return {
                'score': 50,
                'status': 'warning',
                'issues': [f'Model audio tidak ditemukan di {AUDIO_MODEL_PATH} – analisis audio dilewati'],
            }

        try:
            import importlib.util
            current_dir = os.path.dirname(os.path.abspath(__file__))  # → folder utils/
            audio_path  = os.path.join(current_dir, "audio_analyzer.py")

            spec   = importlib.util.spec_from_file_location("audio_analyzer", audio_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.AudioDeepfakeAnalyzer(model_path=AUDIO_MODEL_PATH)
            result   = analyzer.analyze(self.filepath)
            return result
        except Exception as e:
            return {
                'score': 50,
                'status': 'warning',
                'issues': [f'Analisis audio gagal: {str(e)}'],
            }
   
    def _analyze_visual(self) -> dict:
        if not os.path.exists(VISUAL_MODEL_PATH):
            return {
                'score':  50,
                'status': 'warning',
                'issues': [f'Model visual tidak ditemukan di {VISUAL_MODEL_PATH} – analisis visual dilewati'],
            }

        try:
            import importlib.util
            current_dir  = os.path.dirname(os.path.abspath(__file__))
            visual_path  = os.path.join(current_dir, "visual_analyzer.py")

            spec   = importlib.util.spec_from_file_location("visual_analyzer", visual_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            analyzer = module.VisualDeepfakeAnalyzer(model_path=VISUAL_MODEL_PATH)
            result   = analyzer.analyze(self.filepath)
            return result
        except Exception as e:
            return {
                'score':  50,
                'status': 'warning',
                'issues': [f'Analisis visual gagal: {str(e)}'],
            }
   
    def _extract_metadata(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        file_size = os.path.getsize(self.filepath)

        return {
            'fps': round(fps, 2),
            'frame_count': frame_count,
            'resolution': f"{width}x{height}",
            'width': width,
            'height': height,
            'duration_seconds': round(duration, 2),
            'codec': codec_str.strip(),
            'file_size_mb': round(file_size / (1024 * 1024), 2)
        }

    def _extract_sample_frames(self, max_frames=30):
        frames = []
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = min(max_frames, frame_count)
        step = max(1, frame_count // total)

        for i in range(0, frame_count, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
            if len(frames) >= max_frames:
                break

        return frames

    def _calculate_verdict(self, checks):
        weights = {
            'audio':       0.50, 
            'visual':      0.50,
        }

        total_score = 0
        for key, weight in weights.items():
            if key in checks:
                total_score += checks[key]['score'] * weight

        total_score = round(total_score, 1)

        if total_score >= 80:
            verdict = 'AUTHENTIC'
            confidence = 'HIGH' if total_score >= 90 else 'MEDIUM'
        elif total_score >= 55:
            verdict = 'SUSPICIOUS'
            confidence = 'MEDIUM'
        else:
            verdict = 'MANIPULATED'
            confidence = 'HIGH' if total_score < 35 else 'MEDIUM'

        return total_score, verdict, confidence

    def _generate_summary(self, checks):
        all_issues = []
        for key, data in checks.items():
            for issue in data.get('issues', []):
                all_issues.append(issue)

        if not all_issues:
            all_issues = ["No significant anomalies detected"]

        return {
            'total_issues': len([i for i in all_issues if 'No significant' not in i]),
            'issues_found': all_issues
        }
