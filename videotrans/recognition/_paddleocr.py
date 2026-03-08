import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from videotrans.configure import config
from videotrans.recognition._base import BaseRecogn


@dataclass
class PaddleOCRRecogn(BaseRecogn):
    def __post_init__(self):
        super().__post_init__()
        self.media_file = self.audio_file
        self.ocr_fps = max(float(config.settings.get('stt_ocr_fps', 4)), 0.5)
        self.min_conf = max(min(float(config.settings.get('stt_ocr_conf', 0.55)), 1.0), 0.0)
        self.similarity = max(min(float(config.settings.get('stt_ocr_similarity', 0.86)), 1.0), 0.1)
        self.merge_gap_ms = max(int(config.settings.get('stt_ocr_merge_gap_ms', 900)), 100)
        self.min_duration_ms = max(int(config.settings.get('stt_ocr_min_duration_ms', 350)), 100)
        self.roi = self._load_roi()
        self.paddle_lang = self._to_paddle_lang(self.detect_language)

    def _load_roi(self):
        # Store as percentage to support all resolutions.
        raw = config.params.get('stt_ocr_roi')
        if isinstance(raw, dict):
            x1 = float(raw.get('x1', 0.0))
            y1 = float(raw.get('y1', 0.8))
            x2 = float(raw.get('x2', 1.0))
            y2 = float(raw.get('y2', 1.0))
        else:
            x1, y1, x2, y2 = 0.0, 0.8, 1.0, 1.0
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(x1 + 0.01, min(1.0, x2))
        y2 = max(y1 + 0.01, min(1.0, y2))
        return (x1, y1, x2, y2)

    def _to_paddle_lang(self, detect_language):
        if not detect_language or detect_language == 'auto':
            return 'ch'
        lang = detect_language.lower().split('-')[0]
        mapping = {
            'zh': 'ch',
            'en': 'en',
            'ja': 'japan',
            'ko': 'korean',
            'fr': 'fr',
            'de': 'german',
            'es': 'es',
            'ru': 'ru',
            'th': 'th',
            'vi': 'vi',
            'it': 'it',
            'pt': 'pt',
            'ar': 'ar',
            'ta': 'ta',
            'te': 'te',
        }
        return mapping.get(lang, 'ch')

    def _norm(self, text: str) -> str:
        t = text.strip().lower()
        t = re.sub(r'\s+', '', t)
        return t

    def _sim(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    def _extract_text(self, ocr_result):
        if not ocr_result or not ocr_result[0]:
            return '', 0.0
        chunks = []
        confs = []
        for it in ocr_result[0]:
            if not it or len(it) < 2:
                continue
            txt_conf = it[1]
            if not txt_conf or len(txt_conf) < 2:
                continue
            txt = str(txt_conf[0]).strip()
            conf = float(txt_conf[1])
            if txt and conf >= self.min_conf:
                chunks.append(txt)
                confs.append(conf)
        if not chunks:
            return '', 0.0
        return ' '.join(chunks), (sum(confs) / len(confs))

    def _run_ocr(self, ocr, crop):
        try:
            return ocr.ocr(crop, cls=True)
        except Exception as e:
            # Paddle CPU oneDNN can fail on some frame sizes/operators.
            if 'OneDnnContext' in str(e) or 'fused_conv2d' in str(e):
                return ocr.ocr(crop, cls=False)
            raise

    def _exec(self):
        if self._exit():
            return []

        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f'OpenCV not available: {e}')

        try:
            # Import torch first to avoid DLL load ordering issues on some Windows setups.
            try:
                import torch  # noqa: F401
            except Exception:
                pass
            from paddleocr import PaddleOCR
        except Exception as e:
            raise RuntimeError(f'PaddleOCR runtime not available: {e}')

        self._signal(text=f'OCR init lang={self.paddle_lang}, gpu={self.is_cuda}')
        try:
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.paddle_lang,
                use_gpu=bool(self.is_cuda),
                enable_mkldnn=False,
                show_log=False
            )
        except TypeError:
            ocr = PaddleOCR(use_angle_cls=True, lang=self.paddle_lang, use_gpu=bool(self.is_cuda))

        cap = cv2.VideoCapture(self.media_file)
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open media file: {self.media_file}')

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(int(round(video_fps / self.ocr_fps)), 1)
        frame_window_ms = max(int(1000 / max(self.ocr_fps, 1)), 100)

        observations = []
        idx = -1
        while True:
            if self._exit():
                break
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % frame_step != 0:
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self.roi
            lx = max(min(int(w * x1), w - 1), 0)
            ty = max(min(int(h * y1), h - 1), 0)
            rx = max(min(int(w * x2), w), lx + 1)
            by = max(min(int(h * y2), h), ty + 1)
            crop = frame[ty:by, lx:rx]
            if crop.size == 0:
                continue

            try:
                result = self._run_ocr(ocr, crop)
            except Exception:
                continue

            text, conf = self._extract_text(result)
            if not text:
                continue

            ms = int((idx / video_fps) * 1000)
            observations.append({'time_ms': ms, 'text': text, 'conf': conf})

            if len(observations) % 10 == 0:
                self._signal(text=f'OCR scanning... {len(observations)} text frames')

        cap.release()

        if not observations:
            return []

        segments = []
        for ob in observations:
            txt_norm = self._norm(ob['text'])
            if not txt_norm:
                continue
            if not segments:
                segments.append({
                    'start_time': ob['time_ms'],
                    'end_time': ob['time_ms'] + frame_window_ms,
                    'texts': [ob['text']],
                    'last_norm': txt_norm,
                    'conf_sum': ob['conf'],
                    'count': 1,
                })
                continue

            prev = segments[-1]
            sim = self._sim(txt_norm, prev['last_norm'])
            gap = ob['time_ms'] - prev['end_time']

            # Deduplicate near-identical text from adjacent frames.
            if sim >= self.similarity and gap <= self.merge_gap_ms:
                prev['end_time'] = ob['time_ms'] + frame_window_ms
                prev['texts'].append(ob['text'])
                prev['last_norm'] = txt_norm
                prev['conf_sum'] += ob['conf']
                prev['count'] += 1
            else:
                segments.append({
                    'start_time': ob['time_ms'],
                    'end_time': ob['time_ms'] + frame_window_ms,
                    'texts': [ob['text']],
                    'last_norm': txt_norm,
                    'conf_sum': ob['conf'],
                    'count': 1,
                })

        raws = []
        for seg in segments:
            if seg['end_time'] - seg['start_time'] < self.min_duration_ms:
                seg['end_time'] = seg['start_time'] + self.min_duration_ms
            # Choose a representative text (longer usually contains fuller line).
            text = max(seg['texts'], key=lambda x: len(x.strip())).strip()
            if not text:
                continue
            raws.append({
                'start_time': int(seg['start_time']),
                'end_time': int(seg['end_time']),
                'text': text,
            })

        self._signal(text=f'OCR done, segments={len(raws)}')
        return raws

