from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


MODEL_ROOT = Path(__file__).parent.parent / "data" / "anti_spoof_models"
DEFAULT_MODEL_CANDIDATES = [
    MODEL_ROOT / "baseline_resnet18_v1" / "best_model.pt",
    MODEL_ROOT / "display_focus_v2" / "best_model.pt",
]
DEFAULT_REAL_THRESHOLD = 0.50
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MARGIN_RATIO = 0.18


@dataclass
class LivenessPrediction:
    status: str
    real_score: float
    fake_score: float

    @property
    def is_live(self) -> bool:
        return self.status in {"live", "disabled"}


class LivenessDetector:
    def __init__(
        self,
        model_path: Path | None = None,
        real_threshold: float = DEFAULT_REAL_THRESHOLD,
        margin_ratio: float = DEFAULT_MARGIN_RATIO,
    ) -> None:
        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = next((path for path in DEFAULT_MODEL_CANDIDATES if path.exists()), DEFAULT_MODEL_CANDIDATES[0])
        self.real_threshold = float(real_threshold)
        self.margin_ratio = float(margin_ratio)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = DEFAULT_IMAGE_SIZE
        self.enabled = False
        self.error = ""
        self._model: nn.Module | None = None
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self._load_model()

    @property
    def status_text(self) -> str:
        if self.enabled:
            return f"ON ({self.device.type})"
        if self.error:
            return "OFF"
        return "Memuat..."

    def _build_model(self) -> nn.Module:
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.error = f"checkpoint tidak ditemukan: {self.model_path}"
            print(f"[Liveness] {self.error}")
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            self.image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))

            model = self._build_model()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            self._model = model
            self.enabled = True
            self.error = ""
            print(f"[Liveness] model aktif: {self.model_path.name} ({self.device.type})")
        except Exception as exc:
            self._model = None
            self.enabled = False
            self.error = str(exc)
            print(f"[Liveness] gagal load model: {exc}")

    def _crop_face(self, frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = box.astype(int)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None

        margin_x = int(width * self.margin_ratio)
        margin_y = int(height * self.margin_ratio)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame_bgr.shape[1], x2 + margin_x)
        y2 = min(frame_bgr.shape[0], y2 + margin_y)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - self._mean) / self._std
        chw = np.transpose(rgb, (2, 0, 1))
        return torch.from_numpy(chw)

    def predict_many(self, frame_bgr: np.ndarray, boxes: list[np.ndarray]) -> list[LivenessPrediction]:
        if not boxes:
            return []

        if not self.enabled or self._model is None:
            return [
                LivenessPrediction(status="disabled", real_score=1.0, fake_score=0.0)
                for _ in boxes
            ]

        crops: list[torch.Tensor] = []
        valid_indices: list[int] = []
        predictions: list[LivenessPrediction] = [
            LivenessPrediction(status="spoof", real_score=0.0, fake_score=1.0)
            for _ in boxes
        ]

        for idx, box in enumerate(boxes):
            crop = self._crop_face(frame_bgr, np.asarray(box, dtype=np.float32))
            if crop is None:
                continue
            tensor = self._preprocess_crop(crop)
            crops.append(tensor)
            valid_indices.append(idx)

        if not crops:
            return predictions

        batch = torch.stack(crops, dim=0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                logits = self._model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for idx, prob in zip(valid_indices, probs):
            fake_score = float(prob[0])
            real_score = float(prob[1])
            status = "live" if real_score >= self.real_threshold else "spoof"
            predictions[idx] = LivenessPrediction(
                status=status,
                real_score=real_score,
                fake_score=fake_score,
            )
        return predictions
