#!/usr/bin/env python3
"""
Script to extract zero-th elements from eval_results.json files.
Takes a directory as input and searches for files ending with eval_results.json.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict


def find_eval_results_files(directory: str) -> list:
    """Find all files ending with eval_results.json in the given directory."""
    pattern = os.path.join(directory, "**", "*eval_results.json")
    return glob.glob(pattern, recursive=True)


def extract_zero_elements(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the zero-th element from each list value in the JSON data."""
    result = {}
    for key, value in json_data.items():
        if isinstance(value, list) and len(value) > 0:
            result[key] = value[0]
        else:
            # If value is not a list or is empty, keep as is
            result[key] = value
    return result


def process_eval_results_file(file_path: str) -> Dict[str, Any]:
    """Process a single eval_results.json file and return extracted data."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"Warning: {file_path} does not contain a dictionary")
            return {}

        return extract_zero_elements(data)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {file_path}: {e}")
        return {}
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Extract zero-th elements from eval_results.json files")
    parser.add_argument("--directory", "-d", help="Directory to search for eval_results.json files")
    parser.add_argument("--output", "-o", help="Output file path (optional, prints to stdout if not specified)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return 1

    # Find eval_results.json files
    eval_files = find_eval_results_files(args.directory)

    if not eval_files:
        print(f"No eval_results.json files found in {args.directory}")
        return 1

    if args.verbose:
        print(f"Found {len(eval_files)} eval_results.json files:")
        for f in eval_files:
            print(f"  {f}")
        print()

    # Process all files
    all_results = {}
    for file_path in eval_files:
        if args.verbose:
            print(f"Processing: {file_path}")

        result = process_eval_results_file(file_path)
        if result:
            # Use filename as key to avoid conflicts
            filename = os.path.basename(file_path)
            all_results[filename] = result

    print(json.dumps(all_results, indent=2))

    # Output results
    if args.output:
        if not os.path.exists(os.path.dirname(args.output)):
            os.makedirs(os.path.dirname(args.output))

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
