"""
Decorators for runtime stats.
"""

import tracemalloc
import logging
import time
import psutil

log = logging.getLogger(__name__)


def memory_usage(message: str | None = None):
    def _memory_usage(func):
        """
        Decorator to measure memory usage of a function.
        """

        def wrapper(*args, **kwargs):
            tracemalloc.start()
            result = func(*args, **kwargs)
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if message is not None:
                log.info(f"{message}")
            log.info(f"Current memory usage is {current / 1024 / 1024}MB")
            log.info(f"Peak was {peak / 1024 / 1024}MB")
            return result

        return wrapper

    return _memory_usage


def cpu_usage(message: str | None):
    def _cpu_usage(func):
        """
        Decorator to measure CPU usage of a function.
        """

        def wrapper(*args, **kwargs):
            process = psutil.Process()
            start_cpu_times = process.cpu_times()
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            end_cpu_times = process.cpu_times()

            user_cpu_time = end_cpu_times.user - start_cpu_times.user
            system_cpu_time = end_cpu_times.system - start_cpu_times.system
            total_cpu_time = user_cpu_time + system_cpu_time
            elapsed_time = end_time - start_time

            if message is not None:
                log.info(f"{message}")

            log.info(f"User CPU time: {user_cpu_time:.4f} seconds")
            log.info(f"System CPU time: {system_cpu_time:.4f} seconds")
            log.info(f"Total CPU time: {total_cpu_time:.4f} seconds")
            log.info(f"Elapsed time: {elapsed_time:.4f} seconds")

            return result

        return wrapper

    return _cpu_usage


def time_execution(message: str | None = None):
    def _time_execution(func):
        """
        Decorator to measure execution time of a function.
        """

        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_perf_counter = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.time()
            end_perf_counter = time.perf_counter()
            elapsed_time = end_time - start_time
            if message is not None:
                log.info(f"{message}")

            log.info(f"Execution time: {elapsed_time:.4f} seconds")
            log.info(
                f"Performance counter time: {end_perf_counter - start_perf_counter:.4f} seconds"
            )

            return result

        return wrapper

    return _time_execution
