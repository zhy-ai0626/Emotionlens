"""L4 · Speech Coach — timed session with LLM-powered coaching advice (multi-face)."""
import time
import asyncio
from backend.lenses import BaseLens
from backend.config import (
    EMOTION_VA, EMO_CLASSES, L4_POS_V_THRESHOLD, rule_based_advice,
)


class SpeechCoachLens(BaseLens):
    id = "m4"
    title = "Speech Coach"

    def __init__(self):
        super().__init__()
        self._result = None
        self._timeline = []
        self._finalize_started = False  # latched true once server schedules LLM call

    def _on_start(self):
        self._result = None
        self._timeline.clear()
        self._finalize_started = False

    def _on_reset(self):
        self._result = None
        self._timeline.clear()
        self._finalize_started = False

    def _compute_metrics(self) -> dict:
        """Compute metrics from accumulated history (multi-face aware)."""
        all_frames = [h for h in self._history if h["faces"]]
        total = len(all_frames)
        if total == 0:
            return {
                "positivity": 0, "anxiety": 0, "expressiveness": 0,
                "neutral_pct": 0, "timeline": [],
            }

        pos_frames = anx_frames = neu_frames = 0

        for h in all_frames:
            faces = h["faces"]
            # Primary face = largest (likely the speaker)
            primary = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
            probs = primary.get("probs", {})
            # Mean valence across ALL faces for room-level sentiment
            v_sum = sum(self._calc_va(f.get("probs", {}))[0] for f in faces)
            v_mean = v_sum / len(faces)
            dominant = primary.get("dominant", "neutral")

            self._timeline.append({
                "t": round(h["ts"] - self.start_ts, 1) if self.start_ts else 0,
                "valence": round(v_mean, 4),
                "dominant": dominant,
            })

            v_speaker, _ = self._calc_va(probs)
            if v_speaker > L4_POS_V_THRESHOLD:
                pos_frames += 1
            if dominant in ("fear", "sadness") or \
               probs.get("fear", 0) + probs.get("sadness", 0) > 0.5:
                anx_frames += 1
            if dominant == "neutral":
                neu_frames += 1

        positivity = round(pos_frames / total, 4)
        anxiety = round(anx_frames / total, 4)
        neutral_pct = round(neu_frames / total, 4)
        expressiveness = round(1.0 - neutral_pct, 4)

        return {
            "positivity": positivity,
            "anxiety": anxiety,
            "expressiveness": expressiveness,
            "neutral_pct": neutral_pct,
            "timeline": list(self._timeline),
        }

    def get_output(self, ts: float) -> dict:
        out = {
            "state": self.state,
            "duration": self.duration,
            "remaining": round(self.remaining, 1),
        }

        if self.state == "running" and self.remaining <= 0:
            self.state = "generating"
            out["state"] = "generating"
            out["remaining"] = 0

        if self.state == "done" and self._result:
            out["result"] = self._result

        return out

    async def finalize(self):
        """Called after timer expires: compute metrics, call LLM, cache result."""
        metrics = self._compute_metrics()
        from backend.advice import gen_advice
        advice, source = gen_advice(metrics)
        self._result = {**metrics, "advice": advice, "advice_source": source}
        self.state = "done"
