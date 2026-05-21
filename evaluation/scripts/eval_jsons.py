import argparse
import glob
import json
import os
import re

import numpy as np


def is_valid_number(value):
    if isinstance(value, (int, float)):
        return np.isfinite(value)
    return False


def detect_chunks(folder_path):
    """
    Detect chunk files matching: *_<total>-<id>.json
    Returns:
      total_chunks (int): inferred total
      chunk_ids (list[int]): ids found for that total
    """
    # Example filename suffix: _3-0.json
    pattern = os.path.join(folder_path, "*_*-*.json")
    files = glob.glob(pattern)

    # Match the tail part "_<total>-<id>.json"
    rgx = re.compile(r"_(\d+)-(\d+)\.json$")

    totals = {}  # total -> set(ids)
    for fp in files:
        m = rgx.search(fp)
        if not m:
            continue
        total = int(m.group(1))
        cid = int(m.group(2))
        totals.setdefault(total, set()).add(cid)

    if not totals:
        raise FileNotFoundError(
            f"No chunk json files found in {folder_path}. Expected pattern like '*_3-0.json'."
        )

    # If multiple totals exist, pick the one with the most ids (most complete run)
    best_total = max(totals.keys(), key=lambda t: len(totals[t]))
    chunk_ids = sorted(totals[best_total])

    return best_total, chunk_ids


def aggregate_statistics(folder_path, total_chunks=None):
    # auto-detect if not provided
    if total_chunks is None:
        inferred_total, chunk_ids = detect_chunks(folder_path)
        total_chunks = inferred_total
        print(f"[INFO] Auto-detected total_chunks={total_chunks}, found chunk_ids={chunk_ids}")
    else:
        # keep old behavior: assume ids are 0..total_chunks-1
        chunk_ids = list(range(total_chunks))

    # Initialize lists to store data
    distances_to_goal = []
    successes = []
    spls = []
    ndtws = []
    path_lengths = []
    oracle_successes = []
    steps_taken = []
    total_episodes = 0
    invalid_spls = 0
    invalid_distances = 0

    # Iterate over chunk ids
    for cid in chunk_ids:
        pattern = os.path.join(folder_path, f"*_{total_chunks}-{cid}.json")
        file_list = glob.glob(pattern)
        if len(file_list) != 1:
            print(f"Missing or ambiguous chunk file for {total_chunks}-{cid}.json, matched: {len(file_list)}")
            continue
        file_path = file_list[0]

        with open(file_path) as file:
            data = json.load(file)
            # Aggregate data
            for episode_id, episode_data in data.items():
                if is_valid_number(episode_data.get("distance_to_goal")):
                    distances_to_goal.append(episode_data["distance_to_goal"])
                else:
                    invalid_distances += 1

                successes.append(episode_data["success"])

                if is_valid_number(episode_data.get("spl")):
                    spls.append(episode_data["spl"])
                else:
                    invalid_spls += 1

                ndtws.append(episode_data["ndtw"])
                path_lengths.append(episode_data["path_length"])
                oracle_successes.append(episode_data["oracle_success"])
                steps_taken.append(episode_data["steps_taken"])
                total_episodes += 1

    # Calculate statistics
    stats = {
        "mean_distance_to_goal": float(np.mean(distances_to_goal)) if distances_to_goal else "N/A",
        "mean_success": float(np.mean(successes)) if successes else "N/A",
        "mean_spl": float(np.mean(spls)) if spls else "N/A",
        "mean_ndtw": float(np.mean(ndtws)) if ndtws else "N/A",
        "mean_path_length": float(np.mean(path_lengths)) if path_lengths else "N/A",
        "mean_oracle_success": float(np.mean(oracle_successes)) if oracle_successes else "N/A",
        "mean_steps_taken": float(np.mean(steps_taken)) if steps_taken else "N/A",
        "total_episodes": total_episodes,
        "invalid_spls": invalid_spls,
        "invalid_distances": invalid_distances,
        "detected_total_chunks": total_chunks,
        "processed_chunk_ids": chunk_ids,
    }
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate statistics from JSON files.")
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing JSON files.",
    )
    # make total_chunks optional
    parser.add_argument(
        "total_chunks",
        type=int,
        nargs="?",
        default=None,
        help="Total chunks of JSON files to process. If omitted, auto-detect from filenames.",
    )
    args = parser.parse_args()

    statistics = aggregate_statistics(args.folder_path, args.total_chunks)
    print(json.dumps(statistics, indent=4, ensure_ascii=False))