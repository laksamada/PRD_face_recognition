"""
enrollment.py
-------------
Modul 1: Pendaftaran (Enrollment).

EnrollmentSession tidak punya thread internal.
process_frame() dipanggil dari main thread (via Tkinter after loop)
sehingga DML inference selalu di main thread — tidak segfault.
"""

import time
import cv2
import numpy as np
from .face_engine import get_engine
from .database import add_student, is_nim_registered

FRAMES_NEEDED = 5
CAPTURE_INTERVAL_SEC = 0.4
MIN_FACE_CONF = 0.7


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


class EnrollmentSession:
    """
    Sesi pendaftaran tanpa thread internal.
    Caller (main thread) memanggil process_frame() setiap ada frame baru.
    """

    def __init__(self, on_progress: callable, on_done: callable):
        self._on_progress = on_progress   # (count, total, overlay_frame)
        self._on_done = on_done           # (embedding | None)
        self._active = False
        self._embeddings: list[np.ndarray] = []
        self._last_capture = 0.0

    def start(self):
        self._active = True
        self._embeddings = []
        self._last_capture = 0.0

    def cancel(self):
        self._active = False
        self._on_done(None)

    def process_frame(self, frame: np.ndarray):
        """
        Dipanggil dari main thread dengan frame terbaru.
        Melakukan deteksi & ekstraksi embedding jika interval terpenuhi.
        """
        if not self._active:
            return

        overlay = frame.copy()
        count = len(self._embeddings)
        cv2.putText(
            overlay,
            f"Enrollment: {count}/{FRAMES_NEEDED}  Geser kepala pelan...",
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
        )

        now = time.time()
        if now - self._last_capture >= CAPTURE_INTERVAL_SEC:
            app = get_engine()
            faces = app.get(frame)
            if faces and faces[0].det_score >= MIN_FACE_CONF:
                emb = faces[0].normed_embedding
                self._embeddings.append(emb)
                self._last_capture = now

                box = faces[0].bbox.astype(int)
                cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                print(f"  [Enrollment] Sample {len(self._embeddings)}/{FRAMES_NEEDED} "
                      f"(score={faces[0].det_score:.2f})")

        self._on_progress(len(self._embeddings), FRAMES_NEEDED, overlay)

        if len(self._embeddings) >= FRAMES_NEEDED:
            self._active = False
            mean_vec = np.mean(np.array(self._embeddings), axis=0)
            super_vector = _l2_normalize(mean_vec).astype(np.float32)
            self._on_done(super_vector)

    @property
    def is_active(self) -> bool:
        return self._active


def enroll_student_with_embedding(nim: str, name: str, embedding: np.ndarray) -> bool:
    if is_nim_registered(nim):
        print(f"[Enrollment] NIM {nim} sudah terdaftar.")
        return False
    idx = add_student(nim=nim, name=name, embedding=embedding)
    print(f"[Enrollment] Berhasil! {name} ({nim}) disimpan di index {idx}.")
    return True
