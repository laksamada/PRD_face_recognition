"""
app.py  —  Face Attendance  (tulis ulang bersih)
=================================================

Arsitektur 3-thread:

    CameraThread ──► cam_q (maxsize=1)  ──► main thread
                                              │
                                              ├── render ke Canvas (in-place, tanpa buat object baru)
                                              └── kirim copy ke InferenceWorker

    InferenceWorker ──► result_q (maxsize=1) ──► main thread (get_nowait tiap tick)

    LogWorker  ──► queue.Queue (unbounded) ──► tulis CSV di background

Prinsip utama yang menghilangkan FPS drop:

  1. TIDAK ada `self.after(0, callback)` dari background thread ke Tkinter.
     Semua komunikasi thread → main pakai Queue yang di-poll tiap tick.

  2. Render kamera memakai PhotoImage.paste() IN-PLACE.
     Tidak ada objek baru yang dibuat per frame → tidak ada GC pressure.

  3. Inference selalu di InferenceWorker (bukan main thread).
     result_q maxsize=1 → jika main thread lambat, hasil lama langsung dibuang
     dan diganti hasil baru (non-blocking get_nowait).

  4. CSV ditulis di LogWorker → main thread tidak pernah block I/O.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk

from modules.face_engine import get_engine
from modules.matcher import FaceMatcher
from modules.enrollment import enroll_student_with_embedding
from modules.attendance_log import record, get_recorded_today
from modules.database import is_nim_registered

# ── Konstanta ──────────────────────────────────────────────────────────────────
CAMERA_IDX   = 0
W, H         = 640, 480          # ukuran display kamera
TARGET_CAMERA_FPS = 60
INFER_EVERY  = 0.0               # 0 = kirim frame sesering worker siap
TICK_MS      = 16                # ~60 FPS render loop

DEFAULT_CAMERA_BACKENDS = [
    ("DirectShow", cv2.CAP_DSHOW),
    ("MediaFoundation", cv2.CAP_MSMF),
    ("Auto", cv2.CAP_ANY),
]
HIGH_FPS_CAMERA_BACKENDS = [
    ("MediaFoundation", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW),
    ("Auto", cv2.CAP_ANY),
]
CAMERA_READ_FAIL_LIMIT = 20
CAMERA_CORRUPT_LIMIT   = 3
CORRUPT_STD_THRESHOLD  = 60.0
CORRUPT_DIFF_THRESHOLD = 20.0

COLOR_OK      = (0, 210, 0)
COLOR_UNKNOWN = (30, 30, 220)
COLOR_GUIDE   = (0, 230, 230)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def _open_camera():
    errors = []
    backends = HIGH_FPS_CAMERA_BACKENDS if TARGET_CAMERA_FPS >= 60 else DEFAULT_CAMERA_BACKENDS
    for backend_name, backend in backends:
        cap = cv2.VideoCapture(CAMERA_IDX, backend)
        if not cap.isOpened():
            errors.append(f"{backend_name}: open failed")
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, TARGET_CAMERA_FPS)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ready = False
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0 and not _is_probably_corrupt_frame(frame):
                ready = True
                break
            time.sleep(0.03)

        if ready:
            reported_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"[Camera] backend aktif: {backend_name} | req_fps={TARGET_CAMERA_FPS} | reported_fps={reported_fps:.2f}")
            return cap, backend_name

        errors.append(f"{backend_name}: no frames")
        cap.release()

    raise RuntimeError("Tidak bisa membuka kamera: " + ", ".join(errors))


def _is_probably_corrupt_frame(frame: np.ndarray) -> bool:
    f = frame.astype(np.int16)
    std = float(frame.std())
    horiz = float(np.mean(np.abs(f[:, 1:, :] - f[:, :-1, :])))
    vert = float(np.mean(np.abs(f[1:, :, :] - f[:-1, :, :])))
    return (
        std >= CORRUPT_STD_THRESHOLD
        and horiz >= CORRUPT_DIFF_THRESHOLD
        and vert >= CORRUPT_DIFF_THRESHOLD
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Data types
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Face:
    box:   np.ndarray   # [x1, y1, x2, y2]
    label: str
    color: tuple
    nim: Optional[str] = None
    score: float = 0.0
    recognized: bool = False


@dataclass
class InferResult:
    """Satu hasil inference — bisa recognize atau enroll progress."""
    mode:      str                          # "recognize" | "enroll_progress" | "enroll_done"
    faces:     list[Face]        = field(default_factory=list)
    overlay:   Optional[np.ndarray] = None  # untuk enroll, frame yg sudah dianotasi
    embedding: Optional[np.ndarray] = None  # hanya untuk enroll_done
    sample_n:  int = 0                      # jumlah sample terkumpul (enroll)


# ══════════════════════════════════════════════════════════════════════════════
#  CameraThread  —  baca frame, simpan di queue maxsize=1 (always-latest)
# ══════════════════════════════════════════════════════════════════════════════
class CameraThread(threading.Thread):
    def __init__(self, out_q: queue.Queue, stop_ev: threading.Event):
        super().__init__(daemon=True, name="CameraThread")
        self.out_q   = out_q
        self.stop_ev = stop_ev
        self.backend_name = "-"
        self.restart_count = 0
        self.capture_fps = 0.0
        self._fps_n = 0
        self._fps_t = time.perf_counter()

    def _reopen(self, cap, reason: str):
        try:
            cap.release()
        except Exception:
            pass
        self.restart_count += 1
        print(f"[Camera] restart #{self.restart_count}: {reason}")
        time.sleep(0.2)
        cap, self.backend_name = _open_camera()
        return cap

    def run(self):
        cap, self.backend_name = _open_camera()
        read_fail_count = 0
        corrupt_count = 0

        while not self.stop_ev.is_set():
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                read_fail_count += 1
                if read_fail_count >= CAMERA_READ_FAIL_LIMIT:
                    cap = self._reopen(cap, "terlalu banyak frame gagal")
                    read_fail_count = 0
                    corrupt_count = 0
                time.sleep(0.01)
                continue
            read_fail_count = 0

            # Normalise channel format
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))

            if _is_probably_corrupt_frame(frame):
                corrupt_count += 1
                if corrupt_count >= CAMERA_CORRUPT_LIMIT:
                    cap = self._reopen(cap, "stream terdeteksi korup/noise")
                    read_fail_count = 0
                    corrupt_count = 0
                continue
            corrupt_count = 0

            safe_frame = frame.copy()
            self._fps_n += 1
            dt = time.perf_counter() - self._fps_t
            if dt >= 1.0:
                self.capture_fps = self._fps_n / dt
                self._fps_n = 0
                self._fps_t = time.perf_counter()

            # Selalu simpan frame TERBARU — buang yang lama jika penuh
            try:
                self.out_q.get_nowait()
            except queue.Empty:
                pass
            self.out_q.put(safe_frame)

        cap.release()
        print("[Camera] berhenti.")


# ══════════════════════════════════════════════════════════════════════════════
#  InferenceWorker  —  semua InsightFace / DML di sini, TIDAK menyentuh Tkinter
# ══════════════════════════════════════════════════════════════════════════════
class InferenceWorker(threading.Thread):
    ENROLL_STAGES = [
        {
            "name": "Frontal 1",
            "yaw": (-8.0, 8.0),
            "pitch": (-8.0, 8.0),
            "roll_max": 10.0,
            "instruction": "Tatap lurus kamera dan jaga wajah tetap di tengah.",
        },
        {
            "name": "Kiri",
            "yaw": (-32.0, -16.0),
            "pitch": (-12.0, 12.0),
            "roll_max": 12.0,
            "instruction": "Putar kepala ke kiri, jangan menggeser wajah dari tengah.",
        },
        {
            "name": "Frontal 2",
            "yaw": (-8.0, 8.0),
            "pitch": (-8.0, 8.0),
            "roll_max": 10.0,
            "instruction": "Kembali lurus ke depan.",
        },
        {
            "name": "Kanan",
            "yaw": (16.0, 32.0),
            "pitch": (-12.0, 12.0),
            "roll_max": 12.0,
            "instruction": "Putar kepala ke kanan, jangan menggeser wajah dari tengah.",
        },
        {
            "name": "Frontal 3",
            "yaw": (-8.0, 8.0),
            "pitch": (-8.0, 8.0),
            "roll_max": 10.0,
            "instruction": "Tatap lurus sekali lagi lalu tahan sebentar.",
        },
    ]
    ENROLL_MIN_CONF = 0.72
    ENROLL_CAPTURE_INTERVAL = 0.15
    ENROLL_CENTER_TOL = 0.12
    ENROLL_HOLD_SEC = 0.65
    ENROLL_MIN_FACE_RATIO = 0.07
    ENROLL_MIN_BLUR = 85.0

    def __init__(self, task_q: queue.Queue, result_q: queue.Queue,
                 stop_ev: threading.Event):
        super().__init__(daemon=True, name="InferenceWorker")
        self.task_q   = task_q
        self.result_q = result_q
        self.stop_ev  = stop_ev

        self._matcher   = FaceMatcher()
        self._enroll_buf: list[np.ndarray] = []
        self._enrolling  = False
        self._enroll_step = 0
        self._last_enroll_capture = 0.0
        self._enroll_hold_started = 0.0

    # ── Public API (thread-safe) ───────────────────────────────────────────
    def start_enroll(self):
        self._enroll_buf = []
        self._enrolling  = True
        self._enroll_step = 0
        self._last_enroll_capture = 0.0
        self._enroll_hold_started = 0.0

    def cancel_enroll(self):
        self._enroll_buf = []
        self._enrolling  = False
        self._enroll_step = 0
        self._last_enroll_capture = 0.0
        self._enroll_hold_started = 0.0

    def reload_db(self):
        self._matcher.reload()

    @property
    def db_count(self) -> int:
        v = self._matcher._db_vectors
        return v.shape[0] if v is not None else 0

    # ── Event loop ────────────────────────────────────────────────────────
    def run(self):
        engine = get_engine()   # init GPU di thread ini

        while not self.stop_ev.is_set():
            try:
                frame: np.ndarray = self.task_q.get(timeout=0.5)
            except queue.Empty:
                continue

            faces_raw = engine.get(frame)

            if self._enrolling:
                result = self._process_enroll(frame, faces_raw)
            else:
                result = self._process_recognize(faces_raw)

            # Masukkan ke result_q — jika penuh, buang hasil lama (non-blocking)
            try:
                self.result_q.get_nowait()
            except queue.Empty:
                pass
            self.result_q.put(result)

    def _process_recognize(self, faces_raw) -> InferResult:
        faces = []
        for face in faces_raw:
            box = face.bbox.astype(int)
            student, score = self._matcher.find(face.normed_embedding)
            if student:
                label = f"{student['nim']}  {score:.2f}"
                color = COLOR_OK
                nim = student["nim"]
                recognized = True
            else:
                label = f"Unknown {score:.2f}"
                color = COLOR_UNKNOWN
                nim = None
                recognized = False
            faces.append(
                Face(
                    box,
                    label,
                    color,
                    nim=nim,
                    score=score,
                    recognized=recognized,
                )
            )
        return InferResult(mode="recognize", faces=faces)

    def _process_enroll(self, frame: np.ndarray, faces_raw) -> InferResult:
        overlay = frame.copy()
        total_steps = len(self.ENROLL_STAGES)
        n = len(self._enroll_buf)
        step_idx = min(self._enroll_step, total_steps - 1)
        stage = self.ENROLL_STAGES[step_idx]
        stage_name = stage["name"]

        frame_h, frame_w = frame.shape[:2]
        target_px = (frame_w // 2, frame_h // 2)
        radius = int(min(frame_h, frame_w) * 0.11)

        is_aligned = False
        hold_progress = 0.0
        status = stage["instruction"]
        pose_text = "Yaw --.- | Pitch --.- | Roll --.-"

        if len(faces_raw) > 1:
            self._enroll_hold_started = 0.0
            status = "Pastikan hanya satu wajah yang terlihat saat pendaftaran."
        elif faces_raw:
            face = max(
                faces_raw,
                key=lambda item: float(
                    (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                ),
            )
            box = face.bbox.astype(int)
            box = np.array([
                max(box[0], 0),
                max(box[1], 0),
                min(box[2], frame_w - 1),
                min(box[3], frame_h - 1),
            ])
            box_center_x = int((box[0] + box[2]) / 2)
            box_center_y = int((box[1] + box[3]) / 2)
            center_dx = abs((box_center_x / frame_w) - 0.5)
            center_dy = abs((box_center_y / frame_h) - 0.5)
            is_centered = (
                center_dx <= self.ENROLL_CENTER_TOL
                and center_dy <= self.ENROLL_CENTER_TOL
            )

            box_w = max(int(box[2] - box[0]), 1)
            box_h = max(int(box[3] - box[1]), 1)
            face_ratio = (box_w * box_h) / float(frame_w * frame_h)

            face_roi = frame[box[1]:box[3], box[0]:box[2]]
            blur_score = 0.0
            if face_roi.size > 0:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            pose = getattr(face, "pose", None)
            yaw = pitch = roll = 0.0
            pose_ok = False
            if pose is not None and len(pose) == 3:
                pitch, yaw, roll = (float(pose[0]), float(pose[1]), float(pose[2]))
                pose_text = f"Yaw {yaw:+05.1f} | Pitch {pitch:+05.1f} | Roll {roll:+05.1f}"
                pose_ok = (
                    stage["yaw"][0] <= yaw <= stage["yaw"][1]
                    and stage["pitch"][0] <= pitch <= stage["pitch"][1]
                    and abs(roll) <= stage["roll_max"]
                )

            quality_ok = (
                face.det_score >= self.ENROLL_MIN_CONF
                and face_ratio >= self.ENROLL_MIN_FACE_RATIO
                and blur_score >= self.ENROLL_MIN_BLUR
            )
            is_aligned = is_centered and pose_ok and quality_ok

            if is_aligned:
                now = time.time()
                if self._enroll_hold_started == 0.0:
                    self._enroll_hold_started = now
                hold_progress = min(
                    (now - self._enroll_hold_started) / self.ENROLL_HOLD_SEC,
                    1.0,
                )
                remaining = max(
                    self.ENROLL_HOLD_SEC - (now - self._enroll_hold_started),
                    0.0,
                )
                status = f"Pose {stage_name.lower()} pas. Tahan {remaining:.1f} dtk."
                if (
                    hold_progress >= 1.0
                    and (now - self._last_enroll_capture) >= self.ENROLL_CAPTURE_INTERVAL
                ):
                    self._enroll_buf.append(face.normed_embedding)
                    self._enroll_step += 1
                    self._last_enroll_capture = now
                    self._enroll_hold_started = 0.0
                    n = len(self._enroll_buf)
                    status = f"Sample {n}/{total_steps} tersimpan untuk pose {stage_name}."
            else:
                self._enroll_hold_started = 0.0
                if face.det_score < self.ENROLL_MIN_CONF:
                    status = "Wajah belum stabil. Dekatkan wajah dan perbaiki pencahayaan."
                elif face_ratio < self.ENROLL_MIN_FACE_RATIO:
                    status = "Dekatkan wajah ke kamera sampai memenuhi panduan."
                elif blur_score < self.ENROLL_MIN_BLUR:
                    status = "Tahan kepala lebih stabil dulu, gambar masih blur."
                elif not is_centered:
                    status = "Jaga wajah tetap di tengah lingkaran, jangan hanya digeser."
                elif pose is None:
                    status = "Pose wajah belum terbaca, coba hadap kamera lebih jelas."
                else:
                    status = self._pose_guidance(stage, yaw, pitch, roll)

            color = COLOR_OK if is_aligned else (0, 200, 255)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 3)
            cv2.circle(overlay, (box_center_x, box_center_y), 5, color, -1)
            cv2.putText(
                overlay,
                pose_text,
                (10, frame_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
            )

        circle_color = COLOR_OK if is_aligned else COLOR_GUIDE
        cv2.circle(overlay, target_px, radius, circle_color, 2)
        cv2.line(overlay, (target_px[0] - 14, target_px[1]),
                 (target_px[0] + 14, target_px[1]), circle_color, 2)
        cv2.line(overlay, (target_px[0], target_px[1] - 14),
                 (target_px[0], target_px[1] + 14), circle_color, 2)

        bar_x1, bar_y1 = 10, frame_h - 52
        bar_x2, bar_y2 = 220, frame_h - 34
        cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2), (90, 90, 90), 2)
        if hold_progress > 0.0:
            fill_x = max(bar_x1 + 2, int(bar_x1 + (bar_x2 - bar_x1) * hold_progress))
            cv2.rectangle(
                overlay,
                (bar_x1 + 2, bar_y1 + 2),
                (fill_x, bar_y2 - 2),
                COLOR_OK,
                -1,
            )

        cv2.putText(
            overlay,
            f"Langkah {min(self._enroll_step + 1, total_steps)}/{total_steps}: {stage_name}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            COLOR_GUIDE,
            2,
        )
        cv2.putText(
            overlay,
            status,
            (10, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        if len(self._enroll_buf) >= total_steps:
            mean = np.mean(np.array(self._enroll_buf), axis=0)
            norm = np.linalg.norm(mean)
            emb  = (mean / norm).astype(np.float32) if norm > 0 else mean
            self._enroll_buf = []
            self._enrolling  = False
            self._enroll_step = 0
            self._last_enroll_capture = 0.0
            self._enroll_hold_started = 0.0
            return InferResult(mode="enroll_done", overlay=overlay,
                               embedding=emb, sample_n=len(self.ENROLL_STAGES))

        return InferResult(mode="enroll_progress", overlay=overlay, sample_n=n)

    @staticmethod
    def _pose_guidance(stage: dict, yaw: float, pitch: float, roll: float) -> str:
        yaw_min, yaw_max = stage["yaw"]
        pitch_min, pitch_max = stage["pitch"]

        if yaw < yaw_min:
            if yaw_max <= 8.0:
                return "Putar kepala sedikit ke kanan sampai kembali lurus."
            if yaw_max < 0.0:
                return "Terlalu menoleh ke kiri. Balik sedikit ke tengah."
            return "Kurang menoleh ke kanan. Putar kepala sedikit lagi."
        if yaw > yaw_max:
            if yaw_min >= -8.0:
                return "Putar kepala sedikit ke kiri sampai kembali lurus."
            if yaw_min > 0.0:
                return "Terlalu menoleh ke kanan. Balik sedikit ke tengah."
            return "Kurang menoleh ke kiri. Putar kepala sedikit lagi."
        if pitch < pitch_min:
            return "Naikkan posisi dagu sedikit."
        if pitch > pitch_max:
            return "Turunkan posisi dagu sedikit."
        if abs(roll) > stage["roll_max"]:
            return "Luruskan kepala, jangan dimiringkan ke bahu."
        return stage["instruction"]


# ══════════════════════════════════════════════════════════════════════════════
#  LogWorker  —  tulis CSV di background, tidak pernah block main thread
# ══════════════════════════════════════════════════════════════════════════════
class LogWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="LogWorker")
        self._q: queue.Queue = queue.Queue()

    def submit(self, nim: str, name: str, score: float,
               done_cb=None):
        """Non-blocking. done_cb(ok, nim, name) dipanggil setelah tulis selesai."""
        self._q.put((nim, name, score, done_cb))

    def run(self):
        while True:
            nim, name, score, cb = self._q.get()
            ok = record(nim, name, score)
            if cb is not None:
                cb(ok, nim, name)


# ══════════════════════════════════════════════════════════════════════════════
#  VideoCanvas  —  render kamera in-place, ZERO objek baru per frame
# ══════════════════════════════════════════════════════════════════════════════
class VideoCanvas(tk.Canvas):
    """
    ImageTk.PhotoImage dibuat SEKALI saat __init__.
    Tiap frame: cv2.cvtColor + Image.fromarray + PhotoImage.paste().
    paste() update buffer internal PhotoImage langsung — tidak ada alokasi heap.
    """
    def __init__(self, parent, width: int, height: int, **kw):
        bg = kw.pop("bg", "#111122")
        super().__init__(parent, width=width, height=height,
                         bg=bg, highlightthickness=0, **kw)
        self._pil   = Image.new("RGB", (width, height), (15, 15, 30))
        self._photo = ImageTk.PhotoImage(self._pil)
        self._item  = self.create_image(0, 0, anchor="nw", image=self._photo)

    def show(self, frame_bgr: np.ndarray):
        """Tampilkan frame — tidak ada objek baru yang dibuat."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._pil = Image.frombytes("RGB", (frame_bgr.shape[1], frame_bgr.shape[0]),
                                    rgb.tobytes())
        self._photo.paste(self._pil)
        self.itemconfig(self._item, image=self._photo)


# ══════════════════════════════════════════════════════════════════════════════
#  FaceAttendanceApp  —  main thread / Tkinter
# ══════════════════════════════════════════════════════════════════════════════
class FaceAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance")
        self.geometry("900x760")
        self.resizable(False, False)

        # ── State ──────────────────────────────────────────────────────────
        self._stop        = threading.Event()
        self._cam_q:    queue.Queue = queue.Queue(maxsize=1)
        self._task_q:   queue.Queue = queue.Queue(maxsize=1)
        self._result_q: queue.Queue = queue.Queue(maxsize=1)

        self._last_frame:    Optional[np.ndarray] = None
        self._enroll_frame:  Optional[np.ndarray] = None   # overlay dari worker
        self._det_faces:     list[Face]  = []
        self._enrolling      = False
        self._enroll_nim     = ""
        self._enroll_name    = ""
        self._recorded_today: set[str] = get_recorded_today()
        self._pending_logs: set[str] = set()

        # FPS
        self._fps_n    = 0
        self._fps_t    = time.perf_counter()
        self._last_inf = 0.0

        # ── Workers ────────────────────────────────────────────────────────
        self._log    = LogWorker();        self._log.start()
        self._infer  = InferenceWorker(self._task_q, self._result_q, self._stop)
        self._infer.start()
        self._camera = CameraThread(self._cam_q, self._stop)
        self._camera.start()

        # ── UI ─────────────────────────────────────────────────────────────
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._shutdown)
        self._tick()   # mulai loop utama

    # ══════════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        self.tabs = ctk.CTkTabview(self, width=880, height=720)
        self.tabs.pack(padx=10, pady=10)
        self.tabs.add("Presensi")
        self.tabs.add("Pendaftaran")
        self._build_attend()
        self._build_enroll()

    # ── Tab Presensi ───────────────────────────────────────────────────────
    def _build_attend(self):
        tab = self.tabs.tab("Presensi")

        self._attend_canvas = VideoCanvas(tab, W, H)
        self._attend_canvas.pack(pady=(6, 4))

        bar = ctk.CTkFrame(tab, fg_color="transparent")
        bar.pack(fill="x", padx=8, pady=2)

        self._fps_lbl    = ctk.CTkLabel(bar, text="UI FPS: --",
                                        font=("Courier New", 13))
        self._fps_lbl.pack(side="left", padx=8)

        self._status_lbl = ctk.CTkLabel(bar, text="Menunggu wajah...",
                                        font=("Helvetica", 13))
        self._status_lbl.pack(side="left", padx=8)

        self._count_lbl  = ctk.CTkLabel(
            bar, text=f"Hadir: {len(self._recorded_today)}",
            font=("Helvetica", 13))
        self._count_lbl.pack(side="right", padx=8)

        self._cam_lbl = ctk.CTkLabel(
            bar, text="Cam: -",
            font=("Helvetica", 12),
            text_color="gray")
        self._cam_lbl.pack(side="right", padx=8)

    # ── Tab Pendaftaran ────────────────────────────────────────────────────
    def _build_enroll(self):
        tab = self.tabs.tab("Pendaftaran")

        self._enroll_canvas = VideoCanvas(tab, W, H)
        self._enroll_canvas.pack(pady=(6, 4))

        form = ctk.CTkFrame(tab)
        form.pack(pady=4, padx=36, fill="x")

        ctk.CTkLabel(form, text="NIM :", font=("Helvetica", 14)).grid(
            row=0, column=0, padx=10, pady=7, sticky="w")
        self._nim_entry = ctk.CTkEntry(form, placeholder_text="13520001",
                                       width=260)
        self._nim_entry.grid(row=0, column=1, padx=10, pady=7)

        ctk.CTkLabel(form, text="Nama:", font=("Helvetica", 14)).grid(
            row=1, column=0, padx=10, pady=7, sticky="w")
        self._name_entry = ctk.CTkEntry(form, placeholder_text="Nama Lengkap",
                                        width=260)
        self._name_entry.grid(row=1, column=1, padx=10, pady=7)

        btns = ctk.CTkFrame(form, fg_color="transparent")
        btns.grid(row=2, column=0, columnspan=2, pady=8)

        self._start_btn = ctk.CTkButton(btns, text="Mulai Pendaftaran",
                                         command=self._start_enroll,
                                         width=220, height=40)
        self._start_btn.pack(side="left", padx=6)

        self._cancel_btn = ctk.CTkButton(btns, text="Batal",
                                          command=self._cancel_enroll,
                                          width=100, height=40,
                                          fg_color="#883333",
                                          hover_color="#aa4444",
                                          state="disabled")
        self._cancel_btn.pack(side="left", padx=6)

        self._enroll_lbl = ctk.CTkLabel(tab, text="",
                                         font=("Helvetica", 13))
        self._enroll_lbl.pack(pady=3)

        n = self._infer.db_count
        self._db_lbl = ctk.CTkLabel(tab, text=f"Terdaftar: {n} mahasiswa",
                                     font=("Helvetica", 12),
                                     text_color="gray")
        self._db_lbl.pack(pady=2)

    # ══════════════════════════════════════════════════════════════════════
    #  Main tick loop  (berjalan setiap TICK_MS di main thread)
    # ══════════════════════════════════════════════════════════════════════
    def _tick(self):
        now = time.perf_counter()

        # 1. Ambil frame kamera terbaru
        new_frame = False
        try:
            frame = self._cam_q.get_nowait()
            self._last_frame = frame
            new_frame = True
        except queue.Empty:
            frame = self._last_frame

        # 2. Ambil hasil inference terbaru (jika ada)
        try:
            result: InferResult = self._result_q.get_nowait()
            self._handle_result(result)
        except queue.Empty:
            pass

        if frame is not None:
            # 3. Kirim frame ke inference worker jika waktunya DAN ada frame baru
            if new_frame and (now - self._last_inf >= INFER_EVERY):
                if not self._task_q.full():
                    self._task_q.put(frame.copy())
                    self._last_inf = now

            # 4. FPS
            self._fps_n += 1
            dt = now - self._fps_t
            if dt >= 1.0:
                self._fps_lbl.configure(text=f"UI FPS: {self._fps_n / dt:.0f}")
                self._cam_lbl.configure(
                    text=(
                        f"Cam: {self._camera.backend_name} "
                        f"{self._camera.capture_fps:.0f} FPS | Restart: {self._camera.restart_count}"
                    )
                )
                self._fps_n = 0
                self._fps_t = now

            # 5. Render — HANYA tab aktif
            active = self.tabs.get()
            if active == "Presensi":
                self._attend_canvas.show(self._annotated(frame))
            else:
                show = self._enroll_frame if self._enroll_frame is not None \
                    else frame
                self._enroll_canvas.show(show)

        self.after(TICK_MS, self._tick)

    # ══════════════════════════════════════════════════════════════════════
    #  Handle inference result
    # ══════════════════════════════════════════════════════════════════════
    def _handle_result(self, r: InferResult):
        if r.mode == "recognize":
            self._det_faces = r.faces
            recognized_faces = [
                face for face in r.faces
                if face.recognized and face.nim
            ]
            submitted_names: list[str] = []

            for face in recognized_faces:
                nim = face.nim
                if nim is None or nim in self._recorded_today or nim in self._pending_logs:
                    continue

                student = next(
                    (s for s in self._infer._matcher._students if s["nim"] == nim),
                    None,
                )
                if not student:
                    continue

                self._pending_logs.add(nim)
                self._log.submit(
                    student["nim"],
                    student["name"],
                    face.score,
                    done_cb=self._log_done,
                )
                submitted_names.append(student["name"])

            if submitted_names:
                status_text = (
                    f"Terdeteksi: {submitted_names[0]}"
                    if len(submitted_names) == 1
                    else f"Terdeteksi {len(submitted_names)} mahasiswa"
                )
                self._status_lbl.configure(text=status_text, text_color="#cfd8ff")
            elif recognized_faces:
                known_face = recognized_faces[0]
                student = next(
                    (s for s in self._infer._matcher._students if s["nim"] == known_face.nim),
                    None,
                )
                if student:
                    self._status_lbl.configure(
                        text=f"Sudah dikenali: {student['name']}",
                        text_color="#cfcf88",
                    )
            elif r.faces:
                self._status_lbl.configure(text="Tidak dikenali",
                                           text_color="#ee4444")
            else:
                self._status_lbl.configure(text="Menunggu wajah...",
                                           text_color="#ffffff")

        elif r.mode == "enroll_progress":
            self._enroll_frame = r.overlay
            self._enroll_lbl.configure(
                text=f"Mengambil sample {r.sample_n}/{len(InferenceWorker.ENROLL_STAGES)}...",
                text_color="yellow")

        elif r.mode == "enroll_done":
            self._enroll_frame = r.overlay
            self._enrolling    = False
            self._start_btn.configure(state="normal")
            self._cancel_btn.configure(state="disabled")

            ok = enroll_student_with_embedding(
                self._enroll_nim, self._enroll_name, r.embedding)
            if ok:
                self._infer.reload_db()
                self._db_lbl.configure(
                    text=f"Terdaftar: {self._infer.db_count} mahasiswa")
                self._enroll_lbl.configure(
                    text=f"✓ Berhasil: {self._enroll_name} ({self._enroll_nim})",
                    text_color="#00dd00")
                self._nim_entry.delete(0, "end")
                self._name_entry.delete(0, "end")
            else:
                self._enroll_lbl.configure(
                    text=f"✗ NIM {self._enroll_nim} sudah terdaftar.",
                    text_color="#dd4444")
            self._enroll_frame = None

    def _log_done(self, ok: bool, nim: str, name: str):
        """Dipanggil dari LogWorker — schedule UI update ke main thread."""
        # after() hanya dipanggil saat ada absensi baru (~1× per 10 detik per orang)
        # Sama sekali tidak menyebabkan flooding.
        if not self._stop.is_set():
            self.after(0, self._ui_log_done, ok, nim, name)

    def _ui_log_done(self, ok: bool, nim: str, name: str):
        self._pending_logs.discard(nim)
        if ok:
            self._recorded_today.add(nim)
            self._count_lbl.configure(
                text=f"Hadir: {len(self._recorded_today)}")
            self._status_lbl.configure(text=f"✓ Tercatat: {name}",
                                       text_color="#00dd00")

    # ══════════════════════════════════════════════════════════════════════
    #  Draw bounding boxes
    # ══════════════════════════════════════════════════════════════════════
    def _annotated(self, frame: np.ndarray) -> np.ndarray:
        if not self._det_faces:
            return frame
        out = frame.copy()
        for f in self._det_faces:
            x1, y1, x2, y2 = f.box
            cv2.rectangle(out, (x1, y1), (x2, y2), f.color, 2)
            cv2.putText(out, f.label, (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, f.color, 2)
        return out

    # ══════════════════════════════════════════════════════════════════════
    #  Enrollment controls
    # ══════════════════════════════════════════════════════════════════════
    def _start_enroll(self):
        nim  = self._nim_entry.get().strip()
        name = self._name_entry.get().strip()
        if not nim or not name:
            messagebox.showwarning("Input kosong", "Isi NIM dan Nama terlebih dahulu.")
            return
        if is_nim_registered(nim):
            messagebox.showwarning("NIM sudah terdaftar", f"NIM {nim} sudah ada di database.")
            return
        self._enroll_nim  = nim
        self._enroll_name = name
        self._enrolling   = True
        self._infer.start_enroll()
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._enroll_lbl.configure(
            text=f"Ikuti panduan pose kepala untuk {name}...",
            text_color="yellow")
        self.tabs.set("Pendaftaran")

    def _cancel_enroll(self):
        self._enrolling = False
        self._infer.cancel_enroll()
        self._enroll_frame = None
        self._start_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._enroll_lbl.configure(text="Dibatalkan.", text_color="gray")

    # ══════════════════════════════════════════════════════════════════════
    #  Shutdown
    # ══════════════════════════════════════════════════════════════════════
    def _shutdown(self):
        self._stop.set()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = FaceAttendanceApp()
    app.mainloop()
