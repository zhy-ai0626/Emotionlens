"""L2 · CODE RED — real-time threat alert (multi-face aware)."""
import time
import random
from backend.lenses import BaseLens
from backend.config import (
    EMOTION_VA, EMO_CLASSES,
    L2_W1, L2_W2, L2_R_SPIKE, L2_R_HI, L2_SUSTAIN_N, L2_COOLDOWN,
    L2_BANNER_POOL,
)


def _face_r(probs: dict) -> float:
    """Compute risk r for a single face."""
    p_neg = probs.get("anger", 0) + probs.get("disgust", 0)
    v = sum(EMOTION_VA[e]["valence"] * probs.get(e, 0) for e in EMOTION_VA)
    a = sum(EMOTION_VA[e]["arousal"] * probs.get(e, 0) for e in EMOTION_VA)
    return L2_W1 * p_neg + L2_W2 * a


class CodeRedLens(BaseLens):
    id = "m2"
    title = "CODE RED"

    def __init__(self):
        super().__init__()
        self._last_alarm_ts = 0.0

    def get_output(self, ts: float) -> dict:
        empty_dist = {e: 0.0 for e in EMO_CLASSES}
        empty = {
            "risk_level": 0, "alarm": False, "trigger": None,
            "distribution": empty_dist, "banner_text": "", "dominant": "neutral",
        }

        history = self._history
        if not history:
            return empty

        faces = history[-1]["faces"]
        if not faces:
            return empty

        # ── Aggregate distribution across ALL faces ──
        n_faces = len(faces)
        agg_dist = {e: 0.0 for e in EMO_CLASSES}
        dom_counts = {}
        for f in faces:
            probs = f.get("probs", {})
            for e in EMO_CLASSES:
                agg_dist[e] += probs.get(e, 0)
            dom = f.get("dominant", "neutral")
            dom_counts[dom] = dom_counts.get(dom, 0) + 1
        distribution = {e: round(v / n_faces, 4) for e, v in agg_dist.items()}
        dominant = max(dom_counts, key=dom_counts.get) if dom_counts else "neutral"

        # ── Max risk across all faces ──
        max_r = max(_face_r(f.get("probs", {})) for f in faces) if faces else 0

        alarm = False
        trigger = None

        if ts - self._last_alarm_ts > L2_COOLDOWN:
            # ── Spike detection (per-face, trigger if ANY face spikes) ──
            past_frames = [h for h in history if ts - h["ts"] >= 0.4]
            if past_frames and past_frames[-1]["faces"]:
                past_faces = past_frames[-1]["faces"]
                # Match faces by track_id if available, else by size rank
                for f_cur in sorted(faces, key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True):
                    r_cur = _face_r(f_cur.get("probs", {}))
                    # Find closest-matching past face (same track_id or similar position)
                    tid = f_cur.get("track_id")
                    f_past = None
                    if tid is not None:
                        f_past = next((fp for fp in past_faces if fp.get("track_id") == tid), None)
                    if f_past is None:
                        # Fallback: compare with same-size-rank face
                        sorted_past = sorted(past_faces, key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)
                        rank = next((i for i, f2 in enumerate(sorted(faces, key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)) if f2 is f_cur), 0)
                        if rank < len(sorted_past):
                            f_past = sorted_past[rank]
                    if f_past:
                        r_past = _face_r(f_past.get("probs", {}))
                        if r_cur - r_past > L2_R_SPIKE:
                            alarm = True
                            trigger = "spike"
                            break

            # ── Sustained high (ANY face stays above threshold for N frames) ──
            if not alarm:
                recent = [h for h in history if ts - h["ts"] <= 0.5]
                if len(recent) >= L2_SUSTAIN_N:
                    # Check each face independently
                    for face_idx in range(min(len(faces), 3)):  # check top-3 faces
                        f_ref = sorted(faces, key=lambda f: f["bbox"][2] * f["bbox"][3], reverse=True)[face_idx]
                        tid = f_ref.get("track_id")
                        all_high = True
                        for h in recent[-L2_SUSTAIN_N:]:
                            if not h["faces"]:
                                all_high = False; break
                            # Find matching face in this frame
                            if tid is not None:
                                match = next((f2 for f2 in h["faces"] if f2.get("track_id") == tid), None)
                            else:
                                match = h["faces"][face_idx] if face_idx < len(h["faces"]) else None
                            if not match:
                                all_high = False; break
                            fr = _face_r(match.get("probs", {}))
                            if fr < L2_R_HI:
                                all_high = False; break
                        if all_high:
                            alarm = True
                            trigger = "sustained"
                            break

        if alarm:
            self._last_alarm_ts = ts

        return {
            "risk_level": round(max_r, 4),
            "alarm": alarm,
            "trigger": trigger,
            "distribution": distribution,
            "banner_text": random.choice(L2_BANNER_POOL) if alarm else "",
            "dominant": dominant,
        }
