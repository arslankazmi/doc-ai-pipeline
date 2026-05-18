"""Confidence scoring, calibration, and routing logic."""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Threshold (env-configurable)
# ---------------------------------------------------------------------------

EASY_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.85"))


# ---------------------------------------------------------------------------
# Platt calibration wrapper
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Sigmoid (Platt) calibration via sklearn LogisticRegression.

    Usage::

        cal = PlattCalibrator()
        cal.fit(raw_scores, labels)       # labels: 1=correct, 0=wrong
        prob = cal.calibrate(0.73)
        cal.save(Path("calibrator.pkl"))

        # later …
        cal2 = PlattCalibrator()
        cal2.load(Path("calibrator.pkl"))
        prob = cal2.calibrate(0.73)
    """

    def __init__(self) -> None:
        self._model: object | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, raw_scores: list[float], labels: list[int]) -> "PlattCalibrator":
        """Fit a logistic regression on ``(raw_score, correct)`` pairs.

        *labels* must be 1 (prediction was correct) or 0 (wrong).
        Requires sklearn; raises ``ImportError`` if not installed.
        """
        from sklearn.linear_model import LogisticRegression  # type: ignore
        import numpy as np  # type: ignore

        X = np.array(raw_scores, dtype=float).reshape(-1, 1)
        y = np.array(labels, dtype=int)

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(X, y)
        self._model = lr
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calibrate(self, raw_score: float) -> float:
        """Map *raw_score* to a calibrated probability in ``[0, 1]``.

        Returns *raw_score* unchanged if the calibrator has not been fitted.
        """
        if not self.is_fitted:
            return float(raw_score)

        import numpy as np  # type: ignore

        X = np.array([[raw_score]], dtype=float)
        prob: float = float(self._model.predict_proba(X)[0, 1])  # type: ignore[union-attr]
        return prob

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle-serialize the fitted model to *path*."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator is not fitted — nothing to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> "PlattCalibrator":
        """Load a previously saved calibrator from *path*."""
        with open(Path(path), "rb") as fh:
            self._model = pickle.load(fh)
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route(confidence: float) -> str:
    """Return ``"easy"`` if *confidence* >= :data:`EASY_THRESHOLD`, else ``"hard"``."""
    return "easy" if confidence >= EASY_THRESHOLD else "hard"


def aggregate_confidence(field_confidences: dict[str, float]) -> float:
    """Return the minimum value in *field_confidences* (worst-field rule).

    Returns ``0.0`` for an empty dict.
    """
    if not field_confidences:
        return 0.0
    return min(field_confidences.values())
