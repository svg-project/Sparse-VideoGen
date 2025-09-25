import argparse
import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load a jsonl file and return a list of dictionaries.

    Args:
        file_path (str): Path to the jsonl file

    Returns:
        List[Dict]: List of dictionaries from the jsonl file
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data


def calculate_average_density(data: List[Dict], key: str = "density") -> float:
    """
    Calculate the average density from a list of dictionaries.

    Args:
        data (List[Dict]): List of dictionaries containing density information
        key (str): The key in the dictionary that contains the density value

    Returns:
        float: Average density
    """
    densities = [entry[key] for entry in data if key in entry]
    if not densities:
        raise ValueError(f"No '{key}' values found in the data")

    # Remove all densities that are 1.0. They corresponds to warmup part.
    densities = [density for density in densities if density != 1.0]

    return np.mean(densities)


def analyze_density_file(file_path: str, key: str = "density", verbose: bool = True) -> Dict[str, Union[float, int]]:
    """
    Analyze a jsonl file containing density information.

    Args:
        file_path (str): Path to the jsonl file
        key (str): The key in the dictionary that contains the density value
        verbose (bool): Whether to print the results

    Returns:
        Dict[str, Union[float, int]]: Dictionary containing analysis results
    """
    if file_path.endswith(".jsonl"):
        data = load_jsonl(file_path)

        if not data:
            raise ValueError(f"No data found in {file_path}")

        avg_density = calculate_average_density(data, key)

        results = {
            "average_density": avg_density,
            "num_samples": len(data),
            "min_density": min(entry[key] for entry in data if key in entry),
            "max_density": max(entry[key] for entry in data if key in entry),
        }

        if verbose:
            print(f"Analysis of {file_path}:")
            print(f"  Number of samples: {results['num_samples']}")
            print(f"  Average density: {results['average_density']:.4f}")
            print(f"  Min density: {results['min_density']:.4f}")
            print(f"  Max density: {results['max_density']:.4f}")

        return results
    elif file_path.endswith(".txt"):
        densities = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    # Extract density value from line
                    try:
                        density = float(line.strip().split(": ")[1])
                        densities.append(density)
                    except (ValueError, IndexError):
                        continue  # Skip lines that don't match expected format

        if not densities:
            raise ValueError(f"No valid density values found in {file_path}")

        avg_density = np.mean(densities)
        results = {
            "average_density": avg_density,
            "num_samples": len(densities),
            "min_density": min(densities),
            "max_density": max(densities),
        }

        if verbose:
            print(f"Analysis of {file_path}:")
            print(f"  Number of samples: {results['num_samples']}")
            print(f"  Average density: {results['average_density']:.4f}")
            print(f"  Min density: {results['min_density']:.4f}")
            print(f"  Max density: {results['max_density']:.4f}")

        return results
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze density from a jsonl file")
    parser.add_argument("--file_path", "-f", type=str, required=True, help="Path to the jsonl file")
    parser.add_argument(
        "--key", type=str, default="avg_density", help="Key in the json that contains the density value"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save the analysis results (optional)")
    parser.add_argument("--quiet", action="store_true", help="Don't print results to console")

    args = parser.parse_args()

    results = analyze_density_file(args.file_path, args.key, verbose=not args.quiet)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        if not args.quiet:
            print(f"Results saved to {args.output}")
