import argparse
import re
import statistics


def analyze_log(file_path: str, target_number: int):
    """
    Analyze log files, count specific X/X pattern records and calculate average time.

    :param file_path: Path to the log file
    :param target_number: Target number in X/X pattern, e.g., 40
    :return: (count, average time string)
    """
    # Read file content
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Dynamically construct regex to match time patterns like 100%|██████████| 40/40 [mm:ss<
    pattern = re.compile(rf"100%\|██████████\| {target_number}/{target_number} \[(\d{{1,2}}:\d{{2}})<")

    matches = pattern.findall(content)

    # Convert time to seconds
    times_in_seconds = []
    for t in matches:
        minutes, seconds = map(int, t.split(":"))
        times_in_seconds.append(minutes * 60 + seconds)

    count_records = len(matches)
    if times_in_seconds:
        avg_seconds = statistics.mean(times_in_seconds)
        avg_minutes = int(avg_seconds // 60)
        avg_remaining_seconds = int(avg_seconds % 60)
        avg_time = f"{avg_minutes:02d}:{avg_remaining_seconds:02d}"
    else:
        avg_time = "00:00"

    return count_records, avg_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count X/X patterns in log files and calculate average time")
    parser.add_argument("--file", "-f", type=str, help="Path to the log file")
    parser.add_argument(
        "--num_denoising_steps", "-n", type=int, default=50, help="Target number in X/X pattern (default: 40)"
    )
    args = parser.parse_args()

    count, avg_time = analyze_log(args.file, args.num_denoising_steps)
    print(f"Count: {count}")
    print(f"Average time: {avg_time}")
