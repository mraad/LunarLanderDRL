import numpy as np


class Utils:
    @staticmethod
    def decay_schedule(
            start: float,
            final: float,
            decay: float,
            steps: int,
            log_start: float = -2.0) -> np.array:
        decay_steps = int(steps * decay)
        values = np.logspace(log_start, 0, decay_steps)[::-1]
        v_min = values.min()
        v_max = values.max()
        values = (values - v_min) / (v_max - v_min)
        values = final + (start - final) * values
        pad = steps - decay_steps
        values = np.pad(values, (0, pad), 'edge')
        return values