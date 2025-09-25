import os
from contextlib import ContextDecorator

import torch

ENABLE_LOGGING = int(os.getenv("TIME_BENCH", "0")) >= 1
CLEAR_LOG_DATA = int(os.getenv("TIME_BENCH", "0")) == 2


operator_log_data = {}


def clear_operator_log_data():
    operator_log_data.clear()


class TimeLoggingContext(ContextDecorator):
    def __init__(self, operation_type):
        self.operation_type = operation_type
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if ENABLE_LOGGING:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if ENABLE_LOGGING:
            self.end_event.record()
            torch.cuda.synchronize()
            duration = self.start_event.elapsed_time(self.end_event)
            if self.operation_type not in operator_log_data:
                operator_log_data[self.operation_type] = 0
            operator_log_data[self.operation_type] += duration


time_logging_decorator = TimeLoggingContext


def print_operator_log_data(module, input, output):
    if not ENABLE_LOGGING:
        return

    from svg.timer import operator_log_data

    max_key_length = max(len(str(key)) for key in operator_log_data.keys())

    # Sort the operator_log_data by keys
    sorted_operator_log_data = dict(sorted(operator_log_data.items()))
    operator_log_data.clear()
    operator_log_data.update(sorted_operator_log_data)

    # Calculate decimal point alignment
    formatted_lines = []
    for key, value in operator_log_data.items():
        if CLEAR_LOG_DATA:
            # Use milliseconds
            formatted_value = format_aligned_decimal(value)
            line = f"{key:<{max_key_length}} : {formatted_value:>4} ms"
        else:
            # Use seconds
            formatted_value = format_aligned_decimal(value / 1000)
            line = f"{key:<{max_key_length}} : {formatted_value:>4} s"
        formatted_lines.append(line)
    print("\n\n")

    if CLEAR_LOG_DATA:
        clear_operator_log_data()

    # Print all formatted lines
    print("\n".join(formatted_lines))


if __name__ == "__main__":
    x = torch.randn(10000, 10000, device="cuda")

    @time_logging_decorator("example_addition")
    def example_function(x):
        y = x + 1
        return y.cuda()

    @time_logging_decorator("example_multiplication")
    def another_function(x):
        y = x @ x.T
        return y.cuda()

    for i in range(200):
        result = example_function(x)
        result = another_function(x)

    print(operator_log_data)


def format_aligned_decimal(value, max_integer_digits=8, decimal_places=2):
    """Format value, align decimal point"""
    total_width = max_integer_digits + 1 + decimal_places  # integer digits + decimal point + decimal places
    return f"{value:>{total_width}.{decimal_places}f}"
