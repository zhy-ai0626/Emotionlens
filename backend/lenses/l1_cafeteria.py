"""L1 · Cafeteria Mood — timed session with satisfaction analysis."""
import time
from backend.lenses import BaseLens
from backend.config import (
    EMOTION_VA, EMO_CLASSES,
    L1_MIN_FACES, L1_VERDICT_HI, L1_VERDICT_MD, L1_VERDICT_LO,
)


class CafeteriaMoodLens(BaseLens):
    id = "m1"
    title = "Cafeteria Mood"

    def _make_result(self) -> dict:
        """Build the result block from accumulated history."""
        all_faces = [f for h in self._history for f in h["faces"]]
        n = len(all_faces)

        if n < L1_MIN_FACES:
            return {
                "distribution": {e: 0.0 for e in EMO_CLASSES},
                "main_emotion": "neutral",
                "satisfaction": 0,
                "positive_ratio": 0,
                "neutral_ratio": 0,
                "negative_ratio": 0,
                "verdict": "Not enough samples",
                "n_samples": n,
            }

        # Emotion distribution across all faces×frames
        dist = {e: 0.0 for e in EMO_CLASSES}
        v_sum = 0.0
        dom_counts = {e: 0 for e in EMO_CLASSES}
        pos_cnt = neu_cnt = neg_cnt = 0

        for f in all_faces:
            probs = f.get("probs", {})
            for e in EMO_CLASSES:
                dist[e] += probs.get(e, 0)
            v, _ = self._calc_va(probs)
            v_sum += v
            dom = f.get("dominant", "neutral")
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
            if dom in ("happiness", "surprise"):
                pos_cnt += 1
            elif dom == "neutral":
                neu_cnt += 1
            else:
                neg_cnt += 1

        # Normalize distribution
        total_p = sum(dist.values()) or 1.0
        dist = {e: round(v / total_p, 4) for e, v in dist.items()}

        mean_v = v_sum / n
        satisfaction = max(0, min(100, round((mean_v + 1) / 2 * 100)))

        # Verdict
        if satisfaction >= L1_VERDICT_HI:
            verdict = "Very satisfied \U0001f44d"
        elif satisfaction >= L1_VERDICT_MD:
            verdict = "Satisfied"
        elif satisfaction >= L1_VERDICT_LO:
            verdict = "Mixed"
        else:
            verdict = "Unsatisfied"

        return {
            "distribution": dist,
            "main_emotion": max(dom_counts, key=dom_counts.get) if dom_counts else "neutral",
            "satisfaction": satisfaction,
            "positive_ratio": round(pos_cnt / n, 4),
            "neutral_ratio": round(neu_cnt / n, 4),
            "negative_ratio": round(neg_cnt / n, 4),
            "verdict": verdict,
            "n_samples": n,
        }

    def get_output(self, ts: float) -> dict:
        out = {
            "state": self.state,
            "duration": self.duration,
            "remaining": round(self.remaining, 1),
            "n_samples": sum(len(h["faces"]) for h in self._history),
        }

        # Auto-finish when timer expires
        if self.state == "running" and self.remaining <= 0:
            self.state = "done"
            out["state"] = "done"
            out["remaining"] = 0

        if self.state == "done":
            out["result"] = self._make_result()

        return out
