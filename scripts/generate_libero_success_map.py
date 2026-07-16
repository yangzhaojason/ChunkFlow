#!/usr/bin/env python3
"""Generate a success map JSON for Libero dataset.

This script reads the Libero LeRobot dataset and extracts episode-level success
information, saving it as a JSON file that can be used with --data.success-map-path.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
from tqdm import tqdm


def main(repo_id: str, root: str, output_path: str):
    """Extract success flags from Libero dataset and save as JSON."""
    print(f"[Info] Loading dataset: {repo_id} from {root}")

    try:
        dataset = lerobot_dataset.LeRobotDataset(repo_id=repo_id, root=root)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}", file=sys.stderr)
        return 1

    print(f"[Info] Dataset loaded: {len(dataset)} samples")
    print(f"[Info] Extracting episode success information...")

    # Collect unique episodes and their success status
    episode_success = {}

    for i in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[i]
        episode_idx = int(sample["episode_index"])

        # Try to infer success from the data
        # Method 1: Check if there's an explicit success field
        if "success" in sample:
            success = bool(sample["success"])
        # Method 2: Check task completion or other indicators
        # For Libero, we might need to look at the last frame of each episode
        # For now, we'll mark all as successful (you can refine this logic)
        else:
            # Default: assume success if we don't have explicit failure info
            success = True

        # Store the success status (last occurrence wins if episode appears multiple times)
        episode_success[episode_idx] = success

    print(f"[Info] Found {len(episode_success)} unique episodes")
    print(f"[Info] Success rate: {sum(episode_success.values())}/{len(episode_success)} "
          f"({100*sum(episode_success.values())/len(episode_success):.1f}%)")

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(episode_success, f, indent=2)

    print(f"[Info] Saved success map to: {output_path}")
    print(f"\nYou can now use this file with:")
    print(f"  --data.success-map-path={output_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Libero success map JSON")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="physical-intelligence/libero",
        help="Hugging Face repository identity for the Libero dataset",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.environ.get("CHUNKFLOW_LIBERO_DATASET", "datasets/libero_lerobot"),
        help="Local root of the Libero LeRobot dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/libero_success.json",
        help="Output path for the success map JSON file",
    )

    args = parser.parse_args()
    sys.exit(main(args.repo_id, args.root, args.output))
