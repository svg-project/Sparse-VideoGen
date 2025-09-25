from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"
    orange = "\033[38;2;180;60;0m"


def clear_memory_usage():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def print_memory_usage(prefix: str = ""):
    print(
        f"{Color.orange}{prefix} Memory: {torch.cuda.memory_allocated() // 1024 ** 2} / {torch.cuda.max_memory_allocated() // 1024 ** 2} MB{Color.reset}"
    )


def print_args(args):
    print(f"{Color.magenta}Args:{Color.reset}")
    for key, value in args.__dict__.items():
        print(f"{Color.magenta}{key}: {value}{Color.reset}")
