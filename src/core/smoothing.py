from collections import deque, defaultdict
from typing import Optional, Tuple, Deque

class SmoothingBuffer:
    """
    Sliding-window temporal smoothing with confidence-weighted majority and hysteresis.
    Works on tokens (strings) and confidences (0..1).
    """
    def __init__(self, window: int = 9, switch_votes: int = 3, min_conf: float = 0.5, none_token: str = "NONE"):
        self.window = window
        self.switch_votes = switch_votes
        self.min_conf = min_conf
        self.none_token = none_token
        self.buf: Deque[Tuple[str, float]] = deque(maxlen=window)
        self.token: Optional[str] = None
        self.conf: float = 0.0

    def update(self, token: Optional[str], conf: float) -> Tuple[Optional[str], float]:
        if token is None or conf < self.min_conf:
            self.buf.append((self.none_token, 0.0))
            return self.token, self.conf
        self.buf.append((token, conf))
        # Weighted majority
        weights: defaultdict[str, float] = defaultdict(float)
        for t, w in self.buf:
            weights[t] += w
        best_token, best_weight = max(weights.items(), key=lambda x: x[1])
        # Hysteresis: require N votes before switching
        if best_token != self.token:
            votes = sum(1 for t, _ in self.buf if t == best_token)
            if votes >= self.switch_votes:
                self.token = best_token
                total = sum(weights.values()) or 1.0
                self.conf = best_weight / total
        else:
            total = sum(weights.values()) or 1.0
            self.conf = weights.get(self.token, 0.0) / total
        return self.token, self.conf

    def clear(self):
        self.buf.clear()
        self.token = None
        self.conf = 0.0
