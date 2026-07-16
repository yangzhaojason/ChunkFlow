import math
import numbers
import numpy as np
from typing import List, Tuple, Dict, Any


def _as_action_matrix(actions, *, name: str = "actions") -> np.ndarray:
    """Validate and convert an action sequence to finite float64 [T, D]."""
    try:
        array = np.asarray(actions)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real numeric matrix [T, D]") from None
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T, D]")
    if array.shape[1] <= 0:
        raise ValueError(f"{name} action dimension must be positive")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain real numeric actions")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite actions")
    return array


def _row_lp_norm(values: np.ndarray, p: float) -> np.ndarray:
    """Compute row-wise Lp norms without avoidable square under/overflow."""
    absolute = np.abs(values)
    scale = np.max(absolute, axis=1)
    normalized = np.divide(
        absolute,
        scale[:, None],
        out=np.zeros_like(absolute),
        where=scale[:, None] != 0.0,
    )
    factor = np.sum(normalized**p, axis=1) ** (1.0 / p)
    return scale * factor


def _stable_nonnegative_mean(values: np.ndarray) -> float:
    """Average non-negative values without overflowing an intermediate sum."""
    if values.size == 0:
        return 0.0
    scale = float(np.max(values))
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return scale
    return float(scale * np.mean(values / scale))


def compute_msd_delta(actions: np.ndarray, order: int = 1) -> float:
    """
    Compute mean squared difference (MSD) of given derivative order across time.
    actions: [T, D]
    order: 1, 2, or 3 (Δa, Δ^2 a, Δ^3 a)
    """
    actions = _as_action_matrix(actions)
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3")

    diffs = actions
    for _ in range(order):
        diffs = np.diff(diffs, axis=0)
    if diffs.shape[0] == 0:
        return 0.0
    msd = np.mean(np.sum(diffs * diffs, axis=1))
    return float(msd)


def compute_tv_l1(actions: np.ndarray) -> float:
    """
    Total Variation (TV-L1): sum of L1 differences across consecutive time steps, normalized by T-1.
    actions: [T, D]
    """
    actions = _as_action_matrix(actions)
    if actions.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(actions, axis=0))
    return float(np.mean(np.sum(diffs, axis=1)))


def compute_hf_ratio(
    actions: np.ndarray,
    cutoff_ratio: float = 0.3,
) -> float:
    """
    Frequency Regularity: high-frequency energy ratio above a fixed cutoff.
    - actions: [T, D]
    - cutoff_ratio: proportion of Nyquist used as cutoff in [0, 1].
      Example: 0.3 means keep frequencies where f_norm > 0.3.
    Implementation notes:
      - Uses rFFT over time axis for each dimension, averages energy across dims.
      - Excludes DC component (k=0) from the denominator and numerator.
    """
    if not np.isfinite(cutoff_ratio) or not 0.0 <= cutoff_ratio <= 1.0:
        raise ValueError("cutoff_ratio must be finite and within [0, 1]")
    actions = _as_action_matrix(actions)
    T = actions.shape[0]
    if T < 3:
        return 0.0
    x = actions
    scale = float(np.max(np.abs(x)))
    if scale == 0.0:
        return 0.0
    x = x / scale
    # rFFT over time; shape -> [F, D], F = T//2 + 1
    X = np.fft.rfft(x, axis=0)
    # Frequency bins normalized to [0, 1] (Nyquist at 1.0)
    freqs = np.fft.rfftfreq(T, d=1.0)  # d=1 time unit per step
    norm_freqs = freqs / 0.5

    # Exclude DC (k=0)
    X_mag2 = np.abs(X[1:, :]) ** 2
    nf = norm_freqs[1:]
    if X_mag2.size == 0:
        return 0.0

    high_mask = nf >= cutoff_ratio
    num = np.sum(X_mag2[high_mask, :])
    den = float(np.sum(X_mag2))
    if den == 0.0:
        return 0.0
    return float(num / den)


def compute_average_reasoning_latency(
    chunk_latencies,
    executed_action_counts,
) -> float:
    """Return total chunk inference time per executed action."""
    try:
        latencies = np.asarray(chunk_latencies, dtype=object)
        counts = np.asarray(executed_action_counts, dtype=object)
    except (TypeError, ValueError):
        raise ValueError("latencies and action counts must be equal-length vectors") from None
    if latencies.ndim != 1 or counts.ndim != 1 or counts.shape != latencies.shape:
        raise ValueError("latencies and action counts must be equal-length vectors")

    latency_values = []
    for value in latencies.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError("chunk latencies must be real numbers")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("chunk latencies must be finite and non-negative")
        latency_values.append(value)

    count_values = []
    for value in counts.tolist():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError("executed action counts must be integers")
        value = int(value)
        if value < 0:
            raise ValueError("executed action counts must be non-negative")
        count_values.append(value)

    total_actions = sum(count_values)
    if total_actions <= 0:
        raise ValueError("total executed action count must be positive")
    return float(math.fsum(latency_values) / total_actions)


def compute_seam_metrics(
    previous_tail: np.ndarray,
    new_head: np.ndarray,
    p: int = 2,
) -> Dict[str, float]:
    """
    Boundary (Seam) Consistency between overlapped chunks.
    - previous_tail: [O, D] last O actions predicted by previous chunk that would overlap with the next one
    - new_head: [O, D] first O actions predicted by the new chunk
    Returns:
      - bjump: average absolute seam discrepancy across overlap (L2 by default)
      - bratio: seam discrepancy normalized by average intra-chunk L2 step change
    """
    previous_tail = _as_action_matrix(previous_tail, name="previous_tail")
    new_head = _as_action_matrix(new_head, name="new_head")
    if previous_tail.shape != new_head.shape:
        raise ValueError("previous_tail and new_head must have the same shape [O, D]")
    if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(p) or p <= 0:
        raise ValueError("p must be a finite positive norm order")
    O = previous_tail.shape[0]
    if O == 0:
        return {"bjump": 0.0, "bratio": 0.0}

    diff = previous_tail - new_head
    bjump = _stable_nonnegative_mean(_row_lp_norm(diff, p))

    # Reference scale: average intra-chunk L2 step change of the new head region
    if O >= 2:
        intra = np.diff(new_head, axis=0)
        denom = _stable_nonnegative_mean(_row_lp_norm(intra, p))
    else:
        denom = 0.0

    if denom == 0.0:
        bratio = 0.0 if bjump == 0.0 else float("inf")
    else:
        bratio = bjump / denom

    return {"bjump": bjump, "bratio": bratio}


def summarize_trajectory_metrics(
    actions: np.ndarray,
    seam_pairs: List[Tuple[np.ndarray, np.ndarray]] | None = None,
    hf_cutoff_ratio: float = 0.3,
) -> Dict[str, Any]:
    """
    Compute the full set of metrics required by the paper passage:
      - MSD-Δa^1, MSD-Δa^2, MSD-Δa^3 (temporal smoothness)
      - TV-L1 (global variation)
      - HF_ratio (frequency regularity)
      - Bjump, Bratio (boundary seam consistency), averaged across boundaries if provided
    """
    metrics: Dict[str, Any] = {}
    metrics["msd_delta_1"] = compute_msd_delta(actions, order=1)
    metrics["msd_delta_2"] = compute_msd_delta(actions, order=2)
    metrics["msd_delta_3"] = compute_msd_delta(actions, order=3)
    metrics["tv_l1"] = compute_tv_l1(actions)
    metrics["hf_ratio"] = compute_hf_ratio(actions, cutoff_ratio=hf_cutoff_ratio)

    if seam_pairs:
        bjumps, brats = [], []
        for prev_tail, new_head in seam_pairs:
            res = compute_seam_metrics(prev_tail, new_head, p=2)
            bjumps.append(res["bjump"])
            brats.append(res["bratio"])
        metrics["bjump"] = float(np.mean(bjumps)) if bjumps else 0.0
        metrics["bratio"] = float(np.mean(brats)) if brats else 0.0
    else:
        metrics["bjump"] = 0.0
        metrics["bratio"] = 0.0

    return metrics
