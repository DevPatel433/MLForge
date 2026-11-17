"""Learning-rate schedules: callables ``schedule(step) -> multiplier``."""
import math


class StepDecay:
    def __init__(self, drop=0.5, every=100):
        self.drop, self.every = drop, every

    def __call__(self, t):
        return self.drop ** (t // self.every)


class ExponentialDecay:
    def __init__(self, gamma=0.999):
        self.gamma = gamma

    def __call__(self, t):
        return self.gamma ** t


class InverseTimeDecay:
    def __init__(self, decay=0.01):
        self.decay = decay

    def __call__(self, t):
        return 1.0 / (1.0 + self.decay * t)


class CosineAnnealing:
    def __init__(self, t_max=1000, min_factor=0.0):
        self.t_max, self.min_factor = t_max, min_factor

    def __call__(self, t):
        cos = 0.5 * (1 + math.cos(math.pi * min(t, self.t_max) / self.t_max))
        return self.min_factor + (1 - self.min_factor) * cos


class WarmupSchedule:
    """Linear warm-up for ``warmup`` steps, then constant."""

    def __init__(self, warmup=50):
        self.warmup = warmup

    def __call__(self, t):
        return min(1.0, (t + 1) / self.warmup)
