"""L5 · Mimic Game — emotion-mimicking score-attack game."""
import time
import random
from backend.lenses import BaseLens
from backend.config import EMOTION_VA, EMO_CLASSES, L5_HIT_THRESHOLD, L5_TIME_PER_TARGET, L5_TOTAL_ROUNDS, L5_TOTAL_TIME, L5_MAX_SCORE, L5_SCORE_MULTIPLIER


class MimicGameLens(BaseLens):
    id = "m5"
    title = "Mimic Game"

    def __init__(self):
        super().__init__()
        self._target = "happiness"
        self._target_ts = 0.0
        self._score = 0
        self._combo = 0
        self._round = 0
        self._best = 0
        self._game_over = False

    def _on_start(self):
        self._score = 0
        self._combo = 0
        self._round = 0
        self._best = 0
        self._game_over = False
        self._pick_new_target(time.time())

    def _on_reset(self):
        self._on_start()

    def _pick_new_target(self, now: float):
        """Pick a random target emotion (not neutral, not same as current)."""
        candidates = [e for e in EMO_CLASSES if e not in ("neutral", self._target)]
        if not candidates:
            candidates = [e for e in EMO_CLASSES if e != "neutral"]
        self._target = random.choice(candidates)
        self._target_ts = now
        self._round += 1

        # Check game-over conditions
        if self._round > L5_TOTAL_ROUNDS:
            self._game_over = True
        if self.elapsed >= L5_TOTAL_TIME:
            self._game_over = True

    def get_output(self, ts: float) -> dict:
        now = time.time()

        if self.state != "running":
            return {
                "target": self._target,
                "target_emoji": self._target,
                "p_target": 0,
                "score": self._score,
                "combo": self._combo,
                "time_left": L5_TIME_PER_TARGET,
                "round": self._round,
                "game_over": self._game_over,
                "best": self._best,
            }

        # Time since target was set
        elapsed_target = now - self._target_ts
        time_left = max(0, L5_TIME_PER_TARGET - elapsed_target)

        # Get current face probs
        faces = self._history[-1]["faces"] if self._history else []
        p_target = 0.0
        if faces:
            f = max(faces, key=lambda f2: f2["bbox"][2] * f2["bbox"][3])
            p_target = f.get("probs", {}).get(self._target, 0)

        # Check hit (score doubled, capped at L5_MAX_SCORE)
        if p_target > L5_HIT_THRESHOLD:
            gain = (10 + self._combo * 2) * L5_SCORE_MULTIPLIER
            self._score = min(L5_MAX_SCORE, self._score + gain)
            self._combo += 1
            if self._score > self._best:
                self._best = self._score
            if not self._game_over:
                self._pick_new_target(now)
            time_left = L5_TIME_PER_TARGET

        # Check timeout on current target
        if time_left <= 0:
            self._combo = 0  # FAILURE resets combo!
            if not self._game_over:
                self._pick_new_target(now)
            time_left = L5_TIME_PER_TARGET

        # Check total-time game over
        if self.elapsed >= L5_TOTAL_TIME:
            self._game_over = True

        return {
            "target": self._target,
            "target_emoji": self._target,
            "p_target": round(p_target, 4),
            "score": self._score,
            "combo": self._combo,
            "time_left": round(time_left, 1),
            "round": self._round,
            "game_over": self._game_over,
            "best": self._best,
        }
