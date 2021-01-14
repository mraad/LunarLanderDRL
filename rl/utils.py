import numpy as np


class Utils:
    @staticmethod
    def dec_schedule(
            start: float,
            final: float,
            ratio: float,
            steps: int,
            log_start: float = -2.0) -> np.array:
        assert steps > 2, "dec_schedule steps should be greater than 2"
        log_steps = max(2, int(steps * ratio))
        values = np.logspace(log_start, 0, log_steps)[::-1]
        v_min = values.min()
        v_max = values.max()
        values = (values - v_min) / (v_max - v_min)
        values = final + (start - final) * values
        pad = steps - log_steps
        values = np.pad(values, (0, pad), 'edge')
        return values

    @staticmethod
    def inc_schedule(
            start: float,
            final: float,
            ratio: float,
            steps: int,
            log_start: float = -2.0) -> np.array:
        assert steps > 2, "inc_schedule steps should be greater than 2"
        log_steps = max(2, int(steps * ratio))
        values = np.logspace(log_start, 0, log_steps)
        v_min = values.min()
        v_max = values.max()
        values = (values - v_min) / (v_max - v_min)
        values = start + (final - start) * values
        pad = steps - log_steps
        values = np.pad(values, (0, pad), 'edge')
        return values