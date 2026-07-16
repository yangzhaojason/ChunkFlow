import numpy as np
from typing import List, Tuple, Dict, Any


def compute_msd_delta(actions: np.ndarray, order: int = 1) -> float:
    """
    Compute mean squared difference (MSD) of given derivative order across time.
    actions: [T, D]
    order: 1, 2, or 3 (Δa, Δ^2 a, Δ^3 a)
    """
    if actions.ndim != 2:
        raise ValueError("actions must be [T, D]")
    if order not in (1, 2, 3):
        raise ValueError("order must be 1, 2, or 3")

    diffs = actions.astype(np.float64)
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
    if actions.ndim != 2:
        raise ValueError("actions must be [T, D]")
    if actions.shape[0] < 2:
        return 0.0
    diffs = np.abs(np.diff(actions.astype(np.float64), axis=0))
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
    if actions.ndim != 2:
        raise ValueError("actions must be [T, D]")
    T = actions.shape[0]
    if T < 3:
        return 0.0
    x = actions.astype(np.float64)
    # rFFT over time; shape -> [F, D], F = T//2 + 1
    X = np.fft.rfft(x, axis=0)
    # Frequency bins normalized to [0, 1] (Nyquist at 1.0)
    freqs = np.fft.rfftfreq(T, d=1.0)  # d=1 time unit per step
    nyquist = freqs[-1] if freqs[-1] > 0 else 1.0
    norm_freqs = freqs / nyquist

    # Exclude DC (k=0)
    X_mag2 = np.abs(X[1:, :]) ** 2
    nf = norm_freqs[1:]
    if X_mag2.size == 0:
        return 0.0

    high_mask = nf >= cutoff_ratio
    num = np.sum(X_mag2[high_mask, :])
    den = np.sum(X_mag2)
    if den <= 1e-12:
        return 0.0
    return float(num / den)


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
    if previous_tail.shape != new_head.shape:
        raise ValueError("previous_tail and new_head must have the same shape [O, D]")
    O = previous_tail.shape[0]
    if O == 0:
        return {"bjump": 0.0, "bratio": 0.0}

    diff = previous_tail.astype(np.float64) - new_head.astype(np.float64)
    if p == 2:
        bjump = float(np.mean(np.linalg.norm(diff, axis=1)))
    elif p == 1:
        bjump = float(np.mean(np.sum(np.abs(diff), axis=1)))
    else:
        # Generic Lp using powered mean
        bjump = float(np.mean(np.sum(np.abs(diff) ** p, axis=1) ** (1.0 / p)))

    # Reference scale: average intra-chunk L2 step change of the new head region
    if O >= 2:
        intra = np.diff(new_head.astype(np.float64), axis=0)
        if p == 2:
            denom = float(np.mean(np.linalg.norm(intra, axis=1)))
        elif p == 1:
            denom = float(np.mean(np.sum(np.abs(intra), axis=1)))
        else:
            denom = float(np.mean(np.sum(np.abs(intra) ** p, axis=1) ** (1.0 / p)))
        if denom < 1e-12:
            bratio = 0.0
        else:
            bratio = bjump / denom
    else:
        bratio = 0.0

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
