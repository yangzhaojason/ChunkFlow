#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate action chunk smoothness metrics on Calvin dataset frames or model predictions.
- 时间域平滑性: 一阶/二阶/三阶差分 (Δa, Δ²a, Δ³a)
- 边界连续性: chunk 边界跳变幅度 Bjump 及相对跳变 Bratio
- 频域平滑性: 高频能量占比 HF_ratio

python eval_code/eval_action_smoothness.py \
  --pred_dir outputs/predictions/chunkflow \
  --act_seq_len 10 --stride 10 --fc_ratio 0.5 \
  --output outputs/metrics/chunkflow_smoothness.json


python eval_code/eval_action_smoothness.py \
  --pred_dir outputs/predictions/baseline \
  --act_seq_len 10 --stride 10 \
  --output outputs/metrics/baseline_smoothness.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    from policy_models.datasets.utils.episode_utils import lookup_naming_pattern
except Exception:
    lookup_naming_pattern = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action chunk smoothness on Calvin dataset or predictions")
    parser.add_argument("--root_data_dir", type=str, default="", help="Calvin数据根目录（包含 ep_start_end_ids.npy）。pred_dir模式可不填")
    parser.add_argument("--save_format", type=str, default="npz", choices=["npz", "pkl", "npy"], help="帧文件格式（用于GT评测或pred_dir使用逐帧存储时）")
    parser.add_argument("--act_seq_len", type=int, default=10, help="动作chunk长度 L")
    parser.add_argument("--obs_seq_len", type=int, default=1, help="观测序列长度，用于对齐动作开始时刻（GT/逐帧预测模式使用）")
    parser.add_argument("--stride", type=int, default=10, help="采样步长，建议==act_seq_len 以自然对齐chunk")
    parser.add_argument("--max_windows", type=int, default=500000, help="最多计算的窗口数量（用于加速）")
    parser.add_argument("--fc_ratio", type=float, default=0.25, help="频域截止频率占Nyquist比例(0..0.5)")
    parser.add_argument("--output", type=str, default="smooth_report.json", help="输出报告json路径")
    # 预测输入
    parser.add_argument("--pred_dir", type=str, default="", help="模型预测逐序列目录（来自 calvin_evaluate 的 --save_actions_dir，每序列一个npz，键 'pred_actions'）")
    parser.add_argument("--pred_file", type=str, default="", help="模型预测单文件序列(npz/npy/pkl)，含(T,D)或键 'pred_actions'/'actions'")
    parser.add_argument("--pred_start_idx", type=int, default=0, help="pred_file中第0个动作对应的数据集全局帧索引（仅逐帧匹配模式）")
    return parser.parse_args()


def load_ep_indices(root: Path) -> np.ndarray:
    path = root / "ep_start_end_ids.npy"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return np.load(path)


def infer_naming(root: Path, save_format: str) -> Tuple[str, str, int]:
    if lookup_naming_pattern is not None:
        pat, n_digits = lookup_naming_pattern(root, save_format)
        return pat[0], pat[1], n_digits
    candidates = list(root.glob("*." + save_format))[:100]
    if not candidates:
        raise RuntimeError(f"No *.{save_format} files found in {root}")
    import re
    name = candidates[0].name
    m = re.search(r"(.*?)(\d+)(\.[^.]+)$", name)
    if not m:
        raise RuntimeError(f"Cannot infer naming pattern from {name}")
    prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
    return prefix, suffix, len(digits)


def load_action_from_file(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == ".npy":
            arr = np.load(path.as_posix(), allow_pickle=True)
        else:
            data = np.load(path.as_posix(), allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                keys = data.files
                if "rel_actions" in keys:
                    arr = data["rel_actions"]
                elif "actions" in keys:
                    arr = data["actions"]
                else:
                    if len(keys) == 1:
                        arr = data[keys[0]]
                    else:
                        return None
            else:
                arr = data
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return None
        if arr.ndim == 1:
            return arr
        if arr.shape[-1] <= 128:
            return arr.reshape(-1, arr.shape[-1])[0]
        flat = arr.reshape(-1)
        return flat[:128]
    except Exception:
        return None


def load_action_array_file(path: Path) -> np.ndarray:
    data = np.load(path.as_posix(), allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "pred_actions" in data.files:
            arr = data["pred_actions"]
        elif "actions" in data.files:
            arr = data["actions"]
        else:
            if len(data.files) == 1:
                arr = data[data.files[0]]
            else:
                raise KeyError(f"No 'pred_actions' or 'actions' in {path}")
    else:
        arr = data
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise ValueError(f"Prediction array must be 2D (T,D), got shape {arr.shape}")
    return arr.astype(np.float32)


def window_actions(root: Path, prefix: str, suffix: str, n_digits: int,
                   start_idx: int, L: int, obs_seq_len: int) -> Optional[np.ndarray]:
    act_start = start_idx + (obs_seq_len - 1)
    actions: List[np.ndarray] = []
    for i in range(act_start, act_start + L):
        name = f"{prefix}{i:0{n_digits}d}{suffix}"
        path = root / name
        a = load_action_from_file(path)
        if a is None:
            return None
        actions.append(a.astype(np.float32))
    try:
        A = np.stack(actions, axis=0)
        return A
    except Exception:
        return None


def window_actions_pred_array(pred_arr: np.ndarray, start_idx: int, L: int, obs_seq_len: int, pred_start_idx: int) -> Optional[np.ndarray]:
    act_start = start_idx + (obs_seq_len - 1)
    j0 = act_start - pred_start_idx
    if j0 < 0 or j0 + L > pred_arr.shape[0]:
        return None
    return pred_arr[j0:j0 + L]


def diff_metrics(A: np.ndarray) -> Dict[str, float]:
    def l2_mean(x: np.ndarray) -> float:
        return float((x ** 2).sum(axis=-1).mean())
    d1 = np.diff(A, n=1, axis=0)
    d2 = np.diff(A, n=2, axis=0)
    d3 = np.diff(A, n=3, axis=0)
    tv_l1 = float(np.abs(d1).mean())
    return {
        "msd_d1": l2_mean(d1),
        "msd_d2": l2_mean(d2) if d2.shape[0] > 0 else 0.0,
        "msd_d3": l2_mean(d3) if d3.shape[0] > 0 else 0.0,
        "tv_l1": tv_l1,
    }


def boundary_metrics(chunks: List[np.ndarray]) -> Dict[str, float]:
    if len(chunks) < 2:
        return {"bjump": 0.0, "bratio": 0.0}
    bjumps = []
    intra = []
    for k in range(len(chunks) - 1):
        a_end = chunks[k][-1]
        a_start_next = chunks[k + 1][0]
        bjumps.append(float(np.linalg.norm(a_end - a_start_next)))
        d1 = np.diff(chunks[k], n=1, axis=0)
        step = float(np.sqrt((d1 ** 2).sum(axis=-1)).mean() + 1e-8)
        intra.append(step)
    bjump = float(np.mean(bjumps))
    intra_mean = float(np.mean(intra)) if intra else 1.0
    bratio = float(bjump / (intra_mean + 1e-8))
    return {"bjump": bjump, "bratio": bratio}


def high_freq_ratio(A: np.ndarray, fc_ratio: float) -> float:
    L = A.shape[0]
    if L < 4:
        return 0.0
    X = A - A.mean(axis=0, keepdims=True)
    F = np.fft.rfft(X, axis=0)
    P = (F * np.conj(F)).real
    K = P.shape[0]
    cutoff = max(1, int((K - 1) * fc_ratio))
    high = P[cutoff:, :].sum()
    total = P.sum() + 1e-8
    return float(high / total)


def aggregate_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def main():
    args = parse_args()

    # 选择数据来源
    use_pred_dir = len(args.pred_dir) > 0
    use_pred_file = len(args.pred_file) > 0
    use_gt = not use_pred_dir and not use_pred_file

    if use_pred_dir and use_pred_file:
        raise ValueError("请二选一：--pred_dir 或 --pred_file，不要同时指定")

    L = args.act_seq_len
    S = args.stride
    obs_len = args.obs_seq_len

    msd_d1_list: List[float] = []
    msd_d2_list: List[float] = []
    msd_d3_list: List[float] = []
    tv_l1_list: List[float] = []
    hfr_list: List[float] = []
    boundary_chunks: List[np.ndarray] = []
    windows_count = 0

    if use_pred_dir:
        pred_root = Path(args.pred_dir)
        if not pred_root.exists():
            raise FileNotFoundError(f"pred_dir not found: {pred_root}")
        files = sorted(list(pred_root.glob("*.npz")) + list(pred_root.glob("*.npy")))
        if len(files) == 0:
            raise RuntimeError(f"No prediction files (*.npz/*.npy) found in {pred_root}")
        num_sequences = 0
        for f in files:
            try:
                seq = load_action_array_file(f)  # (T,D)
            except Exception as e:
                print(f"Skip {f}: {e}")
                continue
            num_sequences += 1
            T = seq.shape[0]
            t = 0
            while t + L <= T:
                A = seq[t:t + L]
                diffs = diff_metrics(A)
                msd_d1_list.append(diffs["msd_d1"])
                msd_d2_list.append(diffs["msd_d2"])
                msd_d3_list.append(diffs["msd_d3"])
                tv_l1_list.append(diffs["tv_l1"])
                hfr_list.append(high_freq_ratio(A, args.fc_ratio))
                if S == L:
                    boundary_chunks.append(A)
                windows_count += 1
                if windows_count >= args.max_windows:
                    break
                t += S
            if windows_count >= args.max_windows:
                break
        boundary = boundary_metrics(boundary_chunks) if boundary_chunks else {"bjump": 0.0, "bratio": 0.0}
        source = "pred_dir"
        root_str = ""
        pred_dir_str = str(pred_root)
        pred_file_str = ""
        pred_start = None
    elif use_pred_file:
        pred_arr = load_action_array_file(Path(args.pred_file))
        # 使用数据集索引对齐（如果提供了 root_data_dir），否则直接序列滑窗
        if len(args.root_data_dir) > 0:
            root = Path(args.root_data_dir)
            ep_idx = load_ep_indices(root)
            # 需要数据集命名以遍历窗口起点
            prefix, suffix, n_digits = infer_naming(root, args.save_format)
            for (ep_start, ep_end) in ep_idx:
                t = ep_start
                while t + obs_len + L <= ep_end:
                    A = window_actions_pred_array(pred_arr, t, L, obs_len, args.pred_start_idx)
                    if A is not None:
                        diffs = diff_metrics(A)
                        msd_d1_list.append(diffs["msd_d1"])
                        msd_d2_list.append(diffs["msd_d2"])
                        msd_d3_list.append(diffs["msd_d3"])
                        tv_l1_list.append(diffs["tv_l1"])
                        hfr_list.append(high_freq_ratio(A, args.fc_ratio))
                        if S == L:
                            boundary_chunks.append(A)
                        windows_count += 1
                        if windows_count >= args.max_windows:
                            break
                    t += S
                if windows_count >= args.max_windows:
                    break
        else:
            # 仅对单序列滑窗
            T = pred_arr.shape[0]
            t = 0
            while t + L <= T:
                A = pred_arr[t:t + L]
                diffs = diff_metrics(A)
                msd_d1_list.append(diffs["msd_d1"])
                msd_d2_list.append(diffs["msd_d2"])
                msd_d3_list.append(diffs["msd_d3"])
                tv_l1_list.append(diffs["tv_l1"])
                hfr_list.append(high_freq_ratio(A, args.fc_ratio))
                if S == L:
                    boundary_chunks.append(A)
                windows_count += 1
                if windows_count >= args.max_windows:
                    break
                t += S
        boundary = boundary_metrics(boundary_chunks) if boundary_chunks else {"bjump": 0.0, "bratio": 0.0}
        source = "pred_file"
        root_str = args.root_data_dir
        pred_dir_str = ""
        pred_file_str = args.pred_file
        pred_start = args.pred_start_idx
    else:
        # GT 模式
        if len(args.root_data_dir) == 0:
            raise ValueError("GT模式下需要 --root_data_dir")
        root = Path(args.root_data_dir)
        ep_idx = load_ep_indices(root)
        prefix, suffix, n_digits = infer_naming(root, args.save_format)
        for (ep_start, ep_end) in ep_idx:
            t = ep_start
            while t + obs_len + L <= ep_end:
                A = window_actions(root, prefix, suffix, n_digits, t, L, obs_len)
                if A is not None:
                    diffs = diff_metrics(A)
                    msd_d1_list.append(diffs["msd_d1"])
                    msd_d2_list.append(diffs["msd_d2"])
                    msd_d3_list.append(diffs["msd_d3"])
                    tv_l1_list.append(diffs["tv_l1"])
                    hfr_list.append(high_freq_ratio(A, args.fc_ratio))
                    if S == L:
                        boundary_chunks.append(A)
                    windows_count += 1
                    if windows_count >= args.max_windows:
                        break
                t += S
            if windows_count >= args.max_windows:
                break
        boundary = boundary_metrics(boundary_chunks) if boundary_chunks else {"bjump": 0.0, "bratio": 0.0}
        source = "gt"
        root_str = str(root)
        pred_dir_str = ""
        pred_file_str = ""
        pred_start = None

    report = {
        "source": source,
        "root_data_dir": root_str,
        "pred_dir": pred_dir_str,
        "pred_file": pred_file_str,
        "pred_start_idx": pred_start,
        "act_seq_len": L,
        "obs_seq_len": obs_len,
        "stride": S,
        "num_windows": windows_count,
        "metrics": {
            "msd_d1": aggregate_stats(msd_d1_list),
            "msd_d2": aggregate_stats(msd_d2_list),
            "msd_d3": aggregate_stats(msd_d3_list),
            "tv_l1": aggregate_stats(tv_l1_list),
            "hf_ratio": aggregate_stats(hfr_list),
            "boundary": boundary,
        }
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[Smoothness] source={source}  Windows: {windows_count}")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
