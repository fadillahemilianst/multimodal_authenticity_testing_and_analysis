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
                # 'metadata': self._check_metadata_consistency(metadata),
                # 'noise': self._analyze_noise_patterns(frames),
                # 'compression': self._analyze_compression_artifacts(frames),
                # 'temporal': self._analyze_temporal_consistency(frames),
                # 'ela': self._error_level_analysis(frames),
                # 'copy_move': self._detect_copy_move(frames),
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
                'summary': self._generate_summary(checks, verdict, score)
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

    def _check_metadata_consistency(self, metadata):
        issues = []
        score = 100

        # FPS sanity check
        standard_fps = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60]
        fps = metadata['fps']
        if fps <= 0:
            issues.append("Invalid FPS detected (0 or negative)")
            score -= 30
        elif not any(abs(fps - s) < 1 for s in standard_fps):
            issues.append(f"Non-standard FPS: {fps}")
            score -= 15

        # Resolution check
        common_res = [(1920, 1080), (1280, 720), (3840, 2160), (854, 480),
                      (640, 480), (426, 240), (2560, 1440), (4096, 2160)]
        w, h = metadata['width'], metadata['height']
        if not any(r[0] == w and r[1] == h for r in common_res):
            issues.append("Non-standard resolution – may indicate cropping or re-encoding")
            score -= 10

        # Duration vs file size ratio
        if metadata['duration_seconds'] > 0:
            bitrate_kbps = (metadata['file_size_mb'] * 8 * 1024) / metadata['duration_seconds']
            if bitrate_kbps < 50:
                issues.append("Suspiciously low bitrate – possible heavy re-compression")
                score -= 20
            elif bitrate_kbps > 100000:
                issues.append("Unusually high bitrate")
                score -= 5

        return {
            'score': max(0, score),
            'issues': issues,
            'status': 'pass' if score >= 70 else 'warning' if score >= 40 else 'fail'
        }

    def _analyze_noise_patterns(self, frames):
        issues = []
        score = 100

        noise_levels = []
        for frame in frames[:15]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_levels.append(laplacian.var())

        if len(noise_levels) < 2:
            return {'score': 50, 'issues': ['Insufficient frames for analysis'], 'status': 'warning'}

        mean_noise = np.mean(noise_levels)
        std_noise = np.std(noise_levels)
        cv_noise = (std_noise / mean_noise) * 100 if mean_noise > 0 else 0

        # High variation in noise is suspicious
        if cv_noise > 80:
            issues.append("High inconsistency in noise pattern – possible frame splicing")
            score -= 35
        elif cv_noise > 50:
            issues.append("Moderate noise inconsistency detected")
            score -= 15

        # Near-zero noise on many frames suggests AI generation or heavy filtering
        low_noise_frames = sum(1 for n in noise_levels if n < 10)
        ratio = low_noise_frames / len(noise_levels)
        if ratio > 0.6:
            issues.append("Many frames exhibit unusually low noise – may indicate AI generation")
            score -= 30

        return {
            'score': max(0, score),
            'mean_noise': round(float(mean_noise), 4),
            'noise_variation': round(float(cv_noise), 2),
            'issues': issues,
            'status': 'pass' if score >= 70 else 'warning' if score >= 40 else 'fail'
        }

    def _analyze_compression_artifacts(self, frames):
        issues = []
        score = 100

        dct_scores = []
        for frame in frames[:15]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = gray.shape
            # Sample 8x8 DCT blocks
            block_scores = []
            for y in range(0, h - 8, 16):
                for x in range(0, w - 8, 16):
                    block = gray[y:y+8, x:x+8]
                    dct = cv2.dct(block)
                    high_freq = np.abs(dct[4:, 4:]).mean()
                    block_scores.append(high_freq)
            if block_scores:
                dct_scores.append(np.mean(block_scores))

        if not dct_scores:
            return {'score': 50, 'issues': ['Could not analyze compression'], 'status': 'warning'}

        mean_dct = np.mean(dct_scores)
        std_dct = np.std(dct_scores)

        # Very low high-frequency content = over-compressed or AI-generated
        if mean_dct < 0.5:
            issues.append("Extremely low high-frequency content – heavy compression or AI generation suspected")
            score -= 40
        elif mean_dct < 2.0:
            issues.append("Low high-frequency content – possible over-compression")
            score -= 20

        # High variation in compression = multiple sources or editing
        cv_dct = (std_dct / mean_dct * 100) if mean_dct > 0 else 0
        if cv_dct > 60:
            issues.append("Inconsistent compression across frames – possible video editing or splicing")
            score -= 25

        return {
            'score': max(0, score),
            'mean_dct_energy': round(float(mean_dct), 4),
            'compression_variation': round(float(cv_dct), 2),
            'issues': issues,
            'status': 'pass' if score >= 70 else 'warning' if score >= 40 else 'fail'
        }

    def _analyze_temporal_consistency(self, frames):
        issues = []
        score = 100

        if len(frames) < 3:
            return {'score': 50, 'issues': ['Too few frames for temporal analysis'], 'status': 'warning'}

        diffs = []
        for i in range(1, len(frames)):
            prev = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            curr = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
            diff = np.mean(np.abs(curr - prev))
            diffs.append(diff)

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # Detect sudden scene cuts / jumps
        threshold = mean_diff + 3 * std_diff
        abrupt_cuts = sum(1 for d in diffs if d > threshold)

        if abrupt_cuts > len(diffs) * 0.15:
            issues.append(f"Multiple abrupt scene changes detected ({abrupt_cuts} cuts) – possible splicing")
            score -= 35
        elif abrupt_cuts > 2:
            issues.append(f"{abrupt_cuts} abrupt temporal discontinuities found")
            score -= 15

        # Near-zero diffs = static or duplicated frames
        static_frames = sum(1 for d in diffs if d < 0.5)
        if static_frames > len(diffs) * 0.5:
            issues.append("Many duplicate or near-static frames – possible video duplication")
            score -= 20

        return {
            'score': max(0, score),
            'mean_frame_diff': round(float(mean_diff), 4),
            'abrupt_cuts': abrupt_cuts,
            'issues': issues,
            'status': 'pass' if score >= 70 else 'warning' if score >= 40 else 'fail'
        }

    def _error_level_analysis(self, frames):
        """ELA - re-compress and compare to find edited regions"""
        issues = []
        score = 100

        ela_scores = []
        for frame in frames[:10]:
            # Encode to JPEG and re-read
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            recompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            ela = cv2.absdiff(frame, recompressed).astype(np.float32)
            ela_mean = np.mean(ela)
            ela_max = np.max(ela)
            ela_scores.append({'mean': ela_mean, 'max': ela_max})

        if not ela_scores:
            return {'score': 50, 'issues': ['ELA analysis failed'], 'status': 'warning'}

        means = [s['mean'] for s in ela_scores]
        global_mean = np.mean(means)
        global_std = np.std(means)

        # High ELA mean = heavy previous editing
        if global_mean > 15:
            issues.append("High ELA values – indicates multiple rounds of compression/editing")
            score -= 40
        elif global_mean > 8:
            issues.append("Moderate ELA values – some re-compression or editing detected")
            score -= 20

        # High variation in ELA = inconsistent editing across frames
        cv_ela = (global_std / global_mean * 100) if global_mean > 0 else 0
        if cv_ela > 70:
            issues.append("Inconsistent ELA across frames – frame-level editing suspected")
            score -= 25

        return {
            'score': max(0, score),
            'ela_mean': round(float(global_mean), 4),
            'ela_variation': round(float(cv_ela), 2),
            'issues': issues,
            'status': 'pass' if score >= 70 else 'warning' if score >= 40 else 'fail'
        }

    def _detect_copy_move(self, frames):
        """Detect copy-move forgery within frames using feature matching"""
        issues = []
        score = 100

        suspicious_frames = 0
        orb = cv2.ORB_create(nfeatures=500)

        for frame in frames[:8]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray, None)

            if des is None or len(kp) < 10:
                continue

            # Check for duplicate keypoint descriptors (copy-move signature)
            des_list = des.tolist()
            seen = defaultdict(int)
            for d in des_list:
                key = tuple(d[:8])  # Use partial descriptor as hash key
                seen[key] += 1

            duplicates = sum(1 for v in seen.values() if v > 2)
            duplicate_ratio = duplicates / len(des_list) if des_list else 0

            if duplicate_ratio > 0.15:
                suspicious_frames += 1

        ratio = suspicious_frames / min(8, len(frames)) if frames else 0

        if ratio > 0.5:
            issues.append("Copy-move patterns detected in multiple frames – region duplication suspected")
            score -= 45
        elif ratio > 0.25:
            issues.append("Possible copy-move forgery in some frames")
            score -= 20

        return {
            'score': max(0, score),
            'suspicious_frames': suspicious_frames,
            'issues': issues,
            'status': 'pass' if score >= 70 else 'warning' if score >= 40 else 'fail'
        }

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

        # Determine verdict
        critical_fails = sum(1 for k, v in checks.items() if v.get('status') == 'fail')
        warnings = sum(1 for k, v in checks.items() if v.get('status') == 'warning')

        if total_score >= 80 and critical_fails == 0:
            verdict = 'AUTHENTIC'
            confidence = 'HIGH' if total_score >= 90 else 'MEDIUM'
        elif total_score >= 55 and critical_fails <= 1:
            verdict = 'SUSPICIOUS'
            confidence = 'MEDIUM'
        else:
            verdict = 'MANIPULATED'
            confidence = 'HIGH' if total_score < 35 or critical_fails >= 3 else 'MEDIUM'

        return total_score, verdict, confidence

    def _generate_summary(self, checks, verdict, score):
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
