"""Quick checker for AWAC RL signals (rewards/discounts) in the first batch.

Usage:
  uv run scripts/check_awac_signals.py --config-name pi05_libero
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import jax
import jax.numpy as jnp

from openpi.training.chunkflow_batch import PairedChunkBatch
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.transforms import flatten_dict


def _stats(x: Any) -> dict[str, float]:
    x = jnp.asarray(x)
    return {
        "mean": float(jnp.mean(x)),
        "min": float(jnp.min(x)),
        "max": float(jnp.max(x)),
        "nonzero_ratio": float(jnp.mean(jnp.where(x != 0, 1.0, 0.0))),
    }


def main(config_name: str, *, print_raw: bool) -> int:
    cfg = _config.get_config(config_name)
    print(f"[Info] Loaded config: {cfg.name}")

    # Optionally inspect a raw sample before transforms to see available keys
    if print_raw and cfg.data.create(cfg.assets_dirs, cfg.model).rlds_data_dir is None:
        ds = _data_loader.create_torch_dataset(
            cfg.data.create(cfg.assets_dirs, cfg.model),
            cfg.model.action_horizon,
            cfg.model,
        )
        raw = ds[0]
        flat = flatten_dict(raw)
        print("[Raw] Top-level flattened keys (first sample):")
        for k in sorted(flat.keys()):
            v = flat[k]
            try:
                shape = tuple(getattr(v, "shape", ()))
                dtype = getattr(v, "dtype", type(v))
            except Exception:
                shape = ()
                dtype = type(v)
            print(f"  {k}: shape={shape}, dtype={dtype}")

    # Create data loader and fetch a single batch
    loader = _data_loader.create_data_loader(
        cfg,
        sharding=None,  # DataLoader will set a default sharding for JAX framework
        shuffle=False,
        num_batches=1,
        skip_norm_stats=False,
    )
    it = iter(loader)
    try:
        batch = next(it)
    except StopIteration:
        print("[Error] Data loader yielded no batches.")
        return 1
    if isinstance(batch, PairedChunkBatch):
        observation, actions = batch.observation, batch.actions
    else:
        observation, actions = batch

    # Convert to host for printing
    obs = jax.tree.map(jax.device_get, observation)
    acts = jax.device_get(actions)

    print("[Info] Batch shapes:")
    print(f"  state: {tuple(obs.state.shape)}")
    print(f"  actions: {tuple(acts.shape)}")
    for k, img in obs.images.items():
        print(f"  image[{k}]: {tuple(img.shape)}")

    # Check rewards/discounts
    if getattr(obs, "rewards", None) is None:
        print("[Check] rewards: MISSING")
    else:
        r = obs.rewards
        print(f"[Check] rewards: present, shape={tuple(r.shape)}, stats={_stats(r)}")

    if getattr(obs, "discounts", None) is None:
        print("[Check] discounts: MISSING")
    else:
        d = obs.discounts
        print(f"[Check] discounts: present, shape={tuple(d.shape)}, stats={_stats(d)}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="Training config name, e.g., pi05_libero")
    parser.add_argument(
        "--print-raw", action="store_true", help="Print raw dataset keys before transforms (LeRobot datasets)."
    )
    args = parser.parse_args()
    sys.exit(main(args.config_name, print_raw=args.print_raw))
