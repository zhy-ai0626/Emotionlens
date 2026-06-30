"""L3 · Audience Reactions — real-time timeline chart + pie chart (multi-face)."""
import time
from backend.lenses import BaseLens
from backend.config import EMOTION_VA, EMO_CLASSES, L3_DEBOUNCE_K


class AudienceReactionsLens(BaseLens):
    id = "m3"
    title = "Audience Reactions"

    def __init__(self):
        super().__init__()
        self._timeline = []        # [{t, probs:{7}, dominant}]
        self._markers = []         # [{t, emotion}]
        self._cumulative = {e: 0.0 for e in EMO_CLASSES}
        self._total_samples = 0
        self._last_dominant = None
        self._debounce_counter = 0
        self._debounce_candidate = None
        self._start_clock = 0.0
        self._last_snapshot_t = 0.0  # throttle timeline snapshots

    def _on_reset(self):
        self._timeline.clear()
        self._markers.clear()
        self._cumulative = {e: 0.0 for e in EMO_CLASSES}
        self._total_samples = 0
        self._last_dominant = None
        self._debounce_counter = 0
        self._debounce_candidate = None
        self._start_clock = time.time()
        self._last_snapshot_t = 0.0

    def _on_start(self):
        self._start_clock = time.time()

    def get_output(self, ts: float) -> dict:
        now = time.time()
        elapsed = round(now - self._start_clock, 1) if self._start_clock else 0.0

        empty_probs = {e: 0.0 for e in EMO_CLASSES}
        empty_dist = {e: 0.0 for e in EMO_CLASSES}
        if self._total_samples > 0:
            total = sum(self._cumulative.values()) or 1.0
            empty_dist = {e: round(v / total, 4) for e, v in self._cumulative.items()}

        history = self._history
        if not history:
            return {
                "timeline": [], "markers": [], "dominant": "neutral",
                "valence": 0.0, "distribution": empty_dist,
                "elapsed": elapsed, "probs": empty_probs,
            }

        faces = history[-1]["faces"]
        if not faces:
            return {
                "timeline": list(self._timeline), "markers": list(self._markers),
                "dominant": "neutral", "valence": 0.0,
                "distribution": empty_dist, "elapsed": elapsed,
                "probs": empty_probs,
            }

        # Average probs across all faces in this frame
        mean_probs = {e: 0.0 for e in EMO_CLASSES}
        mean_v = 0.0
        n = len(faces)
        for f in faces:
            probs = f.get("probs", {})
            for e in EMO_CLASSES:
                mean_probs[e] += probs.get(e, 0)
            v, _ = self._calc_va(probs)
            mean_v += v
        mean_probs = {e: round(v / n, 4) for e, v in mean_probs.items()}
        mean_v /= n
        dominant = self._dominant_from_probs(mean_probs)

        # Update cumulative distribution
        for e in EMO_CLASSES:
            self._cumulative[e] += mean_probs[e]
        self._total_samples += 1

        # ── Timeline snapshot (throttled to ~2 per second) ──
        if now - self._last_snapshot_t >= 0.5:
            self._last_snapshot_t = now
            self._timeline.append({
                "t": elapsed,
                "probs": dict(mean_probs),
                "dominant": dominant,
            })
            # Keep at most 5 minutes of data
            if len(self._timeline) > 600:
                self._timeline = self._timeline[-600:]

        # ── Debounced dominant-switch markers ──
        if dominant != self._last_dominant:
            if dominant == self._debounce_candidate:
                self._debounce_counter += 1
            else:
                self._debounce_candidate = dominant
                self._debounce_counter = 1
            if self._debounce_counter >= L3_DEBOUNCE_K:
                self._markers.append({"t": elapsed, "emotion": dominant})
                self._last_dominant = dominant
                self._debounce_counter = 0
                self._debounce_candidate = None
        else:
            self._debounce_counter = 0
            self._debounce_candidate = None

        total_cum = sum(self._cumulative.values()) or 1.0
        distribution = {e: round(v / total_cum, 4) for e, v in self._cumulative.items()}

        return {
            "timeline": list(self._timeline),
            "markers": list(self._markers),
            "probs": mean_probs,
            "dominant": dominant,
            "valence": round(mean_v, 4),
            "distribution": distribution,
            "elapsed": elapsed,
        }
