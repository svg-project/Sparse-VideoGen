import argparse
import json
import os
from glob import glob

import numpy as np


def calculate_metrics_mean(jsonl_path):
    """
    Load all data from a jsonl file and calculate the mean of each metric.

    Args:
        jsonl_path (str): Path to the jsonl file

    Returns:
        dict: Dictionary containing the mean values of each metric
    """
    # Check if file exists
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"File not found: {jsonl_path}")

    # Load data from jsonl
    metrics_data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                metrics_data.append(json.loads(line))

    if not metrics_data:
        print(f"No data found in {jsonl_path}")
        return {}

    # Extract all metric keys (excluding idx and seed if present)
    metric_keys = [key for key in metrics_data[0].keys() if key not in ["idx", "seed"]]

    # Calculate mean for each metric
    metrics_mean = {}
    for key in metric_keys:
        values = [data[key] for data in metrics_data if key in data]
        metrics_mean[key] = np.mean(values)

    return metrics_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean metrics from jsonl files")
    parser.add_argument(
        "--jsonl_path", "-j", type=str, required=True, help="Path to the jsonl file or directory containing jsonl files"
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="Path to save the results (default: print to console)"
    )

    args = parser.parse_args()

    # Check if the path is a file or directory
    if os.path.isfile(args.jsonl_path):
        jsonl_files = [args.jsonl_path]
    elif os.path.isdir(args.jsonl_path):
        jsonl_files = glob(os.path.join(args.jsonl_path, "**", "*.jsonl"), recursive=True)
    else:
        raise FileNotFoundError(f"Path not found: {args.jsonl_path}")

    results = {}
    for jsonl_file in jsonl_files:
        print(f"Processing {jsonl_file}...")
        rel_path = os.path.relpath(
            jsonl_file, start=args.jsonl_path if os.path.isdir(args.jsonl_path) else os.path.dirname(args.jsonl_path)
        )
        metrics_mean = calculate_metrics_mean(jsonl_file)
        results[rel_path] = metrics_mean

        # Print results for this file
        print(f"Results for {rel_path}:")
        for metric, value in metrics_mean.items():
            print(f"  {metric:<10}: {value:.3f}")

    # Save results if output path is provided
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_path}")
