#!/usr/bin/env python3
"""
Script to calculate average density across all JSONL files in a directory.

This script uses functions from density.py to process all JSONL files in a given
directory, calculate their individual average densities, and then compute the
overall average density across all files.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from svg.utils.density import analyze_density_file, load_jsonl


def find_jsonl_files(directory: str) -> List[Path]:
    """
    Find all JSONL files in the given directory and its subdirectories.

    Args:
        directory (str): Path to the directory to search

    Returns:
        List[Path]: List of Path objects for all JSONL files found
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all .jsonl files recursively
    jsonl_files = sorted(list(directory_path.rglob("*.jsonl")))
    return jsonl_files


def process_directory_densities(
    directory: str, key: str = "density", verbose: bool = True
) -> Tuple[float, int, List[Dict[str, Union[float, int, str]]]]:
    """
    Process all JSONL files in a directory to calculate average density.

    Args:
        directory (str): Path to the directory containing JSONL files
        key (str): The key in the JSON objects that contains the density value
        verbose (bool): Whether to print progress and results

    Returns:
        Tuple[float, int, List[Dict]]: (overall_average_density, file_count, file_results)
    """
    jsonl_files = find_jsonl_files(directory)

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in directory: {directory}")

    file_results = []
    all_densities = []
    processed_files = 0

    if verbose:
        print(f"Found {len(jsonl_files)} JSONL files in {directory}")
        print("Processing files...")

    for file_path in jsonl_files:
        try:
            if verbose:
                print(f"  Processing: {file_path}")

            # Use the existing analyze_density_file function
            result = analyze_density_file(str(file_path), key, verbose=False)

            # Store results for this file
            file_result = {
                "file_path": str(file_path),
                "average_density": result["average_density"],
                "num_samples": result["num_samples"],
                "min_density": result["min_density"],
                "max_density": result["max_density"],
            }
            file_results.append(file_result)

            # Collect individual densities for overall average calculation
            # We need to load the file again to get individual densities
            data = load_jsonl(str(file_path))
            individual_densities = [entry[key] for entry in data if key in entry]
            # Remove warmup densities (1.0) as done in the original function
            individual_densities = [d for d in individual_densities if d != 1.0]
            all_densities.extend(individual_densities)

            processed_files += 1

        except Exception as e:
            if verbose:
                print(f"  Error processing {file_path}: {e}")
            continue

    if not all_densities:
        raise ValueError("No valid density values found in any files")

    # Calculate overall average density
    overall_average = np.mean(all_densities)

    if verbose:
        print(f"\nProcessed {processed_files} files successfully")
        print(f"Total density samples: {len(all_densities)}")
        print(f"Overall average density: {overall_average:.4f}")

    return overall_average, processed_files, file_results


def save_results(results: Dict[str, Union[float, int, List]], output_path: str, verbose: bool = True) -> None:
    """
    Save the analysis results to a JSON file.

    Args:
        results (Dict): The results dictionary to save
        output_path (str): Path where to save the results
        verbose (bool): Whether to print confirmation message
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Results saved to: {output_path}")


def main():
    """Main function to run the density analysis script."""
    parser = argparse.ArgumentParser(description="Calculate average density across all JSONL files in a directory")
    parser.add_argument("--directory", "-d", type=str, help="Path to the directory containing JSONL files")
    args = parser.parse_args()

    # Process all JSONL files in the directory
    overall_avg, file_count, file_results = process_directory_densities(
        args.directory,
    )

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Directory: {args.directory}")
    print(f"Files processed: {file_count}")
    print(f"Overall average density: {overall_avg:.4f}")


if __name__ == "__main__":
    exit(main())
