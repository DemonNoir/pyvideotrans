from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from videotrans.configure import config
from videotrans.configure.config import tr


class FrameCanvas(QtWidgets.QLabel):
    roiChanged = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background:#111;color:#ddd;border:1px solid #333;")

        self._pixmap = None
        self._frame_size = QtCore.QSize(1, 1)
        self._draw_rect = QtCore.QRect()
        self._roi = {"x1": 0.0, "y1": 0.8, "x2": 1.0, "y2": 1.0}
        self._dragging = False
        self._start = QtCore.QPoint()

    def set_frame(self, qimage):
        self._pixmap = QtGui.QPixmap.fromImage(qimage)
        self._frame_size = qimage.size()
        self._sync_draw_rect_from_roi()
        self.update()

    def set_roi(self, roi):
        self._roi = {
            "x1": max(0.0, min(1.0, float(roi.get("x1", 0.0)))),
            "y1": max(0.0, min(1.0, float(roi.get("y1", 0.8)))),
            "x2": max(0.01, min(1.0, float(roi.get("x2", 1.0)))),
            "y2": max(0.01, min(1.0, float(roi.get("y2", 1.0)))),
        }
        if self._roi["x2"] <= self._roi["x1"]:
            self._roi["x2"] = min(1.0, self._roi["x1"] + 0.01)
        if self._roi["y2"] <= self._roi["y1"]:
            self._roi["y2"] = min(1.0, self._roi["y1"] + 0.01)
        self._sync_draw_rect_from_roi()
        self.roiChanged.emit(self._roi.copy())
        self.update()

    def get_roi(self):
        return self._roi.copy()

    def reset_roi(self):
        self.set_roi({"x1": 0.0, "y1": 0.8, "x2": 1.0, "y2": 1.0})

    def full_frame(self):
        self.set_roi({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0})

    def top_20(self):
        self.set_roi({"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 0.2})

    def bottom_20(self):
        self.set_roi({"x1": 0.0, "y1": 0.8, "x2": 1.0, "y2": 1.0})

    def _frame_rect(self):
        if not self._pixmap:
            return QtCore.QRect(0, 0, self.width(), self.height())
        scaled = self._pixmap.size().scaled(self.size(), QtCore.Qt.KeepAspectRatio)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        return QtCore.QRect(x, y, scaled.width(), scaled.height())

    def _sync_draw_rect_from_roi(self):
        fr = self._frame_rect()
        self._draw_rect = QtCore.QRect(
            fr.left() + int(fr.width() * self._roi["x1"]),
            fr.top() + int(fr.height() * self._roi["y1"]),
            int(fr.width() * (self._roi["x2"] - self._roi["x1"])),
            int(fr.height() * (self._roi["y2"] - self._roi["y1"])),
        )

    def _sync_roi_from_draw_rect(self):
        fr = self._frame_rect()
        if fr.width() <= 0 or fr.height() <= 0:
            return
        rect = self._draw_rect.intersected(fr).normalized()
        if rect.width() < 4 or rect.height() < 4:
            return
        self._roi = {
            "x1": (rect.left() - fr.left()) / fr.width(),
            "y1": (rect.top() - fr.top()) / fr.height(),
            "x2": (rect.right() - fr.left()) / fr.width(),
            "y2": (rect.bottom() - fr.top()) / fr.height(),
        }
        self._roi["x1"] = max(0.0, min(1.0, self._roi["x1"]))
        self._roi["y1"] = max(0.0, min(1.0, self._roi["y1"]))
        self._roi["x2"] = max(0.01, min(1.0, self._roi["x2"]))
        self._roi["y2"] = max(0.01, min(1.0, self._roi["y2"]))
        self.roiChanged.emit(self._roi.copy())

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(event)
        fr = self._frame_rect()
        if not fr.contains(event.pos()):
            return
        self._dragging = True
        self._start = event.pos()
        self._draw_rect = QtCore.QRect(self._start, self._start)
        self.update()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return super().mouseMoveEvent(event)
        self._draw_rect = QtCore.QRect(self._start, event.pos()).normalized()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._dragging:
            self._dragging = False
            self._draw_rect = QtCore.QRect(self._start, event.pos()).normalized()
            self._sync_roi_from_draw_rect()
            self.update()
        return super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._pixmap:
            return
        painter = QtGui.QPainter(self)
        fr = self._frame_rect()
        painter.drawPixmap(fr, self._pixmap)

        if self._draw_rect.isNull() or not self._draw_rect.isValid():
            self._sync_draw_rect_from_roi()

        pen = QtGui.QPen(QtGui.QColor(255, 64, 64), 2)
        painter.setPen(pen)
        painter.drawRect(self._draw_rect)
        painter.fillRect(self._draw_rect, QtGui.QColor(255, 64, 64, 30))


class OCRPreviewDialog(QtWidgets.QDialog):
    def __init__(self, video_path, roi=None, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path).as_posix()
        self._cap = None
        self._fps = 25.0
        self._frame_count = 1
        self._current_index = 0
        self._current_frame = None
        self._ocr = None

        self.setWindowTitle("OCR Preview")
        self.resize(1000, 760)
        self._build_ui()
        self._open_video()
        self.canvas.set_roi(roi or config.params.get("stt_ocr_roi", {}))

    def _run_ocr(self, crop, cls=True):
        ocr = self._ensure_ocr()
        try:
            return ocr.ocr(crop, cls=cls)
        except Exception as e:
            if cls and ('OneDnnContext' in str(e) or 'fused_conv2d' in str(e)):
                return ocr.ocr(crop, cls=False)
            raise

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        row1 = QtWidgets.QHBoxLayout()
        self.video_label = QtWidgets.QLineEdit(self.video_path)
        self.video_label.setReadOnly(True)
        row1.addWidget(self.video_label)
        layout.addLayout(row1)

        self.canvas = FrameCanvas(self)
        self.canvas.roiChanged.connect(self._on_roi_changed)
        layout.addWidget(self.canvas)

        row2 = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("◀")
        self.next_btn = QtWidgets.QPushButton("▶")
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.time_label = QtWidgets.QLabel("00:00.000")
        row2.addWidget(self.prev_btn)
        row2.addWidget(self.next_btn)
        row2.addWidget(self.timeline, 1)
        row2.addWidget(self.time_label)
        layout.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton(tr("Reset") if tr("Reset") else "Reset")
        self.full_btn = QtWidgets.QPushButton("Full Frame")
        self.bottom_btn = QtWidgets.QPushButton("Bottom 20%")
        self.top_btn = QtWidgets.QPushButton("Top 20%")
        self.roi_label = QtWidgets.QLabel("x1=0.000 y1=0.800 x2=1.000 y2=1.000")
        row3.addWidget(self.reset_btn)
        row3.addWidget(self.full_btn)
        row3.addWidget(self.bottom_btn)
        row3.addWidget(self.top_btn)
        row3.addWidget(self.roi_label, 1)
        layout.addLayout(row3)

        row4 = QtWidgets.QHBoxLayout()
        self.test_btn = QtWidgets.QPushButton("OCR Test")
        self.scan_btn = QtWidgets.QPushButton("Quick Scan (10)")
        self.ocr_label = QtWidgets.QLabel("text= ; conf=0.000")
        row4.addWidget(self.test_btn)
        row4.addWidget(self.scan_btn)
        row4.addWidget(self.ocr_label, 1)
        layout.addLayout(row4)

        row5 = QtWidgets.QHBoxLayout()
        self.marker_list = QtWidgets.QListWidget()
        self.marker_list.setMaximumHeight(110)
        row5.addWidget(self.marker_list)
        layout.addLayout(row5)

        row6 = QtWidgets.QHBoxLayout()
        row6.addStretch(1)
        self.ok_btn = QtWidgets.QPushButton("OK")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        row6.addWidget(self.ok_btn)
        row6.addWidget(self.cancel_btn)
        layout.addLayout(row6)

        self.prev_btn.clicked.connect(lambda: self._seek(self._current_index - 1))
        self.next_btn.clicked.connect(lambda: self._seek(self._current_index + 1))
        self.timeline.valueChanged.connect(self._seek)
        self.reset_btn.clicked.connect(self.canvas.reset_roi)
        self.full_btn.clicked.connect(self.canvas.full_frame)
        self.bottom_btn.clicked.connect(self.canvas.bottom_20)
        self.top_btn.clicked.connect(self.canvas.top_20)
        self.test_btn.clicked.connect(self._ocr_test)
        self.scan_btn.clicked.connect(self._quick_scan)
        self.marker_list.itemDoubleClicked.connect(self._jump_marker)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def _open_video(self):
        import cv2

        self._cap = cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(max(0, self._frame_count - 1))
        self._seek(0)

    def _frame_to_image(self, frame):
        import cv2

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

    def _seek(self, index):
        if self._cap is None:
            return
        index = max(0, min(int(index), self._frame_count - 1))
        if index == self._current_index and self._current_frame is not None:
            return
        self._cap.set(1, index)
        ok, frame = self._cap.read()
        if not ok:
            return
        self._current_index = index
        self._current_frame = frame
        self.canvas.set_frame(self._frame_to_image(frame))
        self.timeline.blockSignals(True)
        self.timeline.setValue(index)
        self.timeline.blockSignals(False)
        ms = int((index / self._fps) * 1000)
        self.time_label.setText(self._ms_to_text(ms))

    def _ms_to_text(self, ms):
        s = ms // 1000
        mm = s // 60
        ss = s % 60
        mmm = ms % 1000
        return f"{mm:02d}:{ss:02d}.{mmm:03d}"

    def _on_roi_changed(self, roi):
        self.roi_label.setText(
            f"x1={roi['x1']:.3f} y1={roi['y1']:.3f} x2={roi['x2']:.3f} y2={roi['y2']:.3f}"
        )

    def _ensure_ocr(self):
        if self._ocr is not None:
            return self._ocr
        # Import torch first to avoid DLL load ordering issues on some Windows setups.
        try:
            import torch  # noqa: F401
        except Exception:
            pass
        from paddleocr import PaddleOCR

        lang_code = str(config.params.get("stt_ocr_lang", "ch"))
        use_gpu = bool(config.params.get("cuda", config.params.get("stt_cuda", False)))
        try:
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang_code,
                use_gpu=use_gpu,
                enable_mkldnn=False,
                show_log=False
            )
        except TypeError:
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang_code, use_gpu=use_gpu)
        return self._ocr

    def _crop_current(self):
        if self._current_frame is None:
            return None
        h, w = self._current_frame.shape[:2]
        roi = self.canvas.get_roi()
        x1 = max(0, min(w - 1, int(w * roi["x1"])))
        y1 = max(0, min(h - 1, int(h * roi["y1"])))
        x2 = max(x1 + 1, min(w, int(w * roi["x2"])))
        y2 = max(y1 + 1, min(h, int(h * roi["y2"])))
        return self._current_frame[y1:y2, x1:x2]

    def _ocr_test(self):
        crop = self._crop_current()
        if crop is None or crop.size == 0:
            self.ocr_label.setText("text= ; conf=0.000")
            return
        try:
            rs = self._run_ocr(crop, cls=True)
            txts = []
            confs = []
            if rs and rs[0]:
                for row in rs[0]:
                    if row and len(row) > 1 and row[1]:
                        txts.append(str(row[1][0]).strip())
                        confs.append(float(row[1][1]))
            txt = " ".join([t for t in txts if t])
            conf = (sum(confs) / len(confs)) if confs else 0.0
            self.ocr_label.setText(f"text={txt[:120]} ; conf={conf:.3f}")
        except Exception as e:
            self.ocr_label.setText(f"OCR error: {e}")

    def _quick_scan(self):
        if self._cap is None:
            return
        self.marker_list.clear()
        total = max(1, self._frame_count)
        points = [int((total - 1) * i / 9) for i in range(10)]
        hit_count = 0
        for idx in points:
            self._seek(idx)
            crop = self._crop_current()
            if crop is None or crop.size == 0:
                continue
            try:
                rs = self._run_ocr(crop, cls=True)
            except Exception:
                continue
            if not rs or not rs[0]:
                continue
            first = rs[0][0][1][0] if rs[0][0] and len(rs[0][0]) > 1 else ""
            text = str(first).strip()
            if text:
                hit_count += 1
                ms = int((idx / self._fps) * 1000)
                item = QtWidgets.QListWidgetItem(f"{self._ms_to_text(ms)} | {text[:80]}")
                item.setData(QtCore.Qt.UserRole, idx)
                self.marker_list.addItem(item)
        if hit_count == 0:
            self.marker_list.addItem("No text found in quick scan")

    def _jump_marker(self, item):
        idx = item.data(QtCore.Qt.UserRole)
        if idx is None:
            return
        self._seek(int(idx))

    def closeEvent(self, event):
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)

    def get_roi(self):
        return self.canvas.get_roi()

