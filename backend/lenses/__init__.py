"""EmotionLens · Lens base class.

Each lens = one "application" consuming the shared emotion stream.
To add a new application: subclass BaseLens, implement get_output(), register in LENS_REGISTRY.
"""
import time


class BaseLens:
    """Base class for all emotion-analysis lenses."""

    id: str = ""
    title: str = ""

    def __init__(self):
        self.state = "idle"        # idle | running | done | generating
        self.duration = 0          # selected duration in seconds
        self.start_ts = 0.0        # absolute time when "running" began (time.time())
        self._history = []         # [{ts, faces}]

    # ── Control protocol ──────────────────────────────────────

    def handle_control(self, action: str, duration: int | None = None):
        """Handle start/stop/reset control messages from frontend."""
        if action == "start":
            self.state = "running"
            self.start_ts = time.time()
            if duration is not None:
                self.duration = duration
            self._history.clear()
            self._on_start()
        elif action == "stop":
            self.state = "idle"
            self._on_stop()
        elif action == "reset":
            self.state = "idle"
            self.duration = 0
            self.start_ts = 0.0
            self._history.clear()
            self._on_reset()

    def _on_start(self): pass
    def _on_stop(self): pass
    def _on_reset(self): pass

    # ── Frame feeding ─────────────────────────────────────────

    def add_frame(self, ts: float, faces: list):
        """Store a frame for this lens's private buffer."""
        self._history.append({"ts": ts, "faces": faces})
        # Cleanup old entries (> 600s)
        self._history = [h for h in self._history if ts - h["ts"] < 600]

    # ── Output (override in subclasses) ────────────────────────

    def get_output(self, ts: float) -> dict:
        """Return the lens-specific mode_output dict for the frontend."""
        raise NotImplementedError

    # ── Helpers ────────────────────────────────────────────────

    @staticmethod
    def _calc_va(probs: dict) -> tuple[float, float]:
        """Compute valence & arousal from emotion probabilities."""
        from backend.config import EMOTION_VA
        v = sum(EMOTION_VA[e]["valence"] * probs.get(e, 0) for e in EMOTION_VA)
        a = sum(EMOTION_VA[e]["arousal"] * probs.get(e, 0) for e in EMOTION_VA)
        return v, a

    @staticmethod
    def _dominant_from_probs(probs: dict) -> str:
        """Get the dominant emotion from a probability dict."""
        if not probs:
            return "neutral"
        return max(probs, key=probs.get)

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since start (0 if idle)."""
        if self.state not in ("running", "done", "generating"):
            return 0.0
        return time.time() - self.start_ts

    @property
    def remaining(self) -> float:
        """Seconds remaining in countdown."""
        if self.state not in ("running", "done", "generating"):
            return 0.0
        return max(0.0, self.duration - self.elapsed)
