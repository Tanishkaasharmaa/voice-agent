"""Silero VAD using ONNX Runtime.

The ONNX model file lives in the models directory (downloaded during setup
with user consent — never silently). The detector state machine is in
``vad.base``; this class only turns audio frames into probabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vad.base import VADEngine

# 16 kHz, 512-sample frames (32 ms) as the Silero ONNX model expects.
FRAME_SIZE = 512
SAMPLE_RATE = 16000

_DEFAULT_MODEL_NAME = "silero_vad.onnx"
_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"


class SileroVAD(VADEngine):
    def __init__(self, model_path=None) -> None:
        # Normalize to a Path — callers may pass a str (default_model_path
        # used to return one), and is_available() calls .exists() on it.
        self.model_path = Path(model_path) if model_path is not None else None
        self._session = None
        self._state: np.ndarray | None = None
        # Last 64 samples of the previous frame — the v5 ONNX graph expects
        # them PREPENDED to every 512-sample frame (576-sample input).
        self._context = np.zeros(64, dtype=np.float32)

    def is_available(self) -> bool:
        return self.model_path is not None and self.model_path.exists()

    def load(self) -> None:
        if self._session is not None:
            return
        if not self.is_available():
            raise FileNotFoundError(
                f"Silero VAD model not found at {self.model_path}. "
                "Use the Setup dialog to download it."
            )
        import onnxruntime as ort  # type: ignore

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._reset_state()

    def _reset_state(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)

    def probability(self, frame) -> float:
        """frame: numpy float32 array of 512 samples at 16 kHz."""
        if self._session is None:
            self.load()
        if not isinstance(frame, np.ndarray):
            frame = np.asarray(frame, dtype=np.float32)
        frame = np.asarray(frame, dtype=np.float32).ravel()
        if len(frame) < FRAME_SIZE:
            frame = np.pad(frame, (0, FRAME_SIZE - len(frame)))
        frame = frame[:FRAME_SIZE].astype(np.float32, copy=False)
        # v5 model input: [64-sample context || 512 new samples].
        input_ = np.concatenate([self._context, frame]).reshape(1, -1)
        outputs = self._session.run(
            None,
            {
                "input": input_,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        prob = float(outputs[0].ravel()[0])
        self._state = outputs[1]
        self._context = frame[-64:].copy()
        return prob

    def reset(self) -> None:
        self._reset_state()


def default_model_path(models_dir) -> Path:
    return models_dir / _DEFAULT_MODEL_NAME
