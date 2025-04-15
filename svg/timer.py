import os
import time

import torch
import functools
from contextlib import ContextDecorator

ENABLE_LOGGING = int(os.getenv("TIME_BENCH", "0")) >= 3


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

if __name__ == "__main__":
    x = torch.randn(10000, 10000, device='cuda')

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

