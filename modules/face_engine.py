"""
face_engine.py  —  Singleton InsightFace dengan GPU (DML) → CPU fallback.
"""
from __future__ import annotations

from insightface.app import FaceAnalysis
import onnxruntime as ort

_engine: FaceAnalysis | None = None
_ALLOWED_MODULES = ["detection", "recognition", "landmark_3d_68"]


def get_engine(det_size: tuple[int, int] = (640, 640)) -> FaceAnalysis:
    global _engine
    if _engine is None:
        providers = (
            ["DmlExecutionProvider", "CPUExecutionProvider"]
            if "DmlExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        print(f"[Engine] providers: {providers}")
        print(f"[Engine] allowed modules: {_ALLOWED_MODULES}")
        _engine = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=_ALLOWED_MODULES,
        )
        _engine.prepare(ctx_id=0, det_size=det_size)
    return _engine
